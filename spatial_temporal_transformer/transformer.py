from tqdm import tqdm
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from spatial_temporal_transformer.vqvae import SpatialSelfAttention, attention

MIN_TEMPORAL_UNITS = 10
MAX_TEMPORAL_UNITS = 20


def cal_kl_loss(mean, logvar):
    return -0.5 * (1 + logvar - jnp.square(mean) - jnp.exp(logvar))


def positional_embeddings(time, depth):
    depth = depth // 2

    positions = time[:, None]
    depths = jnp.arange(depth)[None, :] / depth

    rads = positions * (1 / 10000 ** depths)
    embeddings = jnp.concatenate([jnp.sin(rads), jnp.cos(rads)], axis=-1)
    return embeddings


def create_mask(seq_len, mask):
    m = jnp.logical_not(mask.astype(jnp.bool))[:, None, None, :]
    cm = jnp.triu(jnp.ones((seq_len, seq_len)), 1).astype(jnp.bool)[None, None, :, :]
    m = jnp.logical_or(m, cm)
    m = m.astype(jnp.float32)
    m = jnp.where(m == 1.0, jnp.full(m.shape, -200000.0), m)
    return m


class FeedForward(nn.Module):
    dropout: float

    @nn.compact
    def __call__(self, x, det):
        C = x.shape[-1]
        x = nn.swish(nn.Dense(C * 2)(x))
        x = nn.Dropout(self.dropout, deterministic=det)(x)
        x = nn.Dense(C)(x)
        return x


class PixelSTAttention(nn.Module):
    heads: int
    dropout: float

    @nn.compact
    def __call__(self, h, mask, deterministic):
        h = nn.LayerNorm()(h)

        B, N, HW, C = h.shape
        cph = C // self.heads

        x = nn.Dense(3 * C, use_bias=False)(h)
        q, k, v = jnp.split(x.reshape(B, N, HW * self.heads, 3 * cph).transpose(0, 2, 1, 3), 3, axis=-1)

        mask = create_mask(N, mask)
        op = attention(q, k, v, cph, mask=mask)

        op = op.transpose(0, 2, 1, 3).reshape(B, N, HW, C) + h
        op = FeedForward(self.dropout)(op, deterministic) + op

        return op


class TransformerDecoder(nn.Module):
    heads: int
    n_layers: int
    embed_dim: int
    norm_g: int
    dropout: float

    @nn.compact
    def __call__(self, q, mask, deterministic):
        B, T, H, W, C = q.shape

        kl_loss = 0.0

        q = q.reshape(B * T, H, W, C)

        q = nn.Conv(self.embed_dim, (3, 3), padding='SAME')(q)

        for n in range(self.n_layers):
            q = SpatialSelfAttention(self.heads, self.norm_g)(q)
            q = q.reshape(B, T, H * W, self.embed_dim)

            td = positional_embeddings(jnp.arange(T), self.embed_dim)
            td = nn.swish(nn.Dense(self.embed_dim)(td))[None, :, None, :]
            q += td

            q = PixelSTAttention(self.heads, self.dropout)(q, mask, deterministic)
            q = q.reshape(B * T, H, W, self.embed_dim)

            q = nn.Conv(self.embed_dim * 2, (3, 3), padding='SAME')(q)
            q, (mean, logvar) = self.gaussian_sampling(q)
            q = nn.Conv(self.embed_dim, (3, 3), padding='SAME')(q)

            kl_loss += cal_kl_loss(mean, logvar).mean()

        kl_loss = kl_loss / self.n_layers

        q = q.reshape(B, T, H, W, self.embed_dim)
        q = nn.Conv(self.embed_dim, (3, 3), padding='SAME')(q)
        return q, kl_loss

    def gaussian_sampling(self, x):
        mean, logvar = jnp.split(x, 2, -1)
        logvar = jnp.clip(logvar, -30.0, 20.0)
        eps = jr.normal(self.make_rng("normal"), mean.shape)
        sampled = mean + jnp.exp(logvar * 0.5) * eps
        return sampled, (mean, logvar)


class SpatialTemporalTransformer(nn.Module):
    c_out: int
    heads: int
    n_layers: int
    embed_dim: int
    norm_g: int
    dropout: float

    vq_instance: nn.Module

    def setup(self) -> None:
        self.vq_model = self.vq_instance

        self.decoder = TransformerDecoder(self.heads, self.n_layers, self.embed_dim, self.norm_g, self.dropout)

        self.end = nn.Sequential([
            nn.GroupNorm(self.norm_g),
            nn.Conv(self.c_out, (3, 3), padding='SAME')
        ])

    @nn.compact
    def __call__(self, x, mask, deterministic):
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C)

        z = self.vq_model.encode_vt(x)
        BT, LH, LW, LC = z.shape
        z = z.reshape(B, T, LH, LW, LC)

        SOS = jr.normal(self.make_rng('normal'), (B, 1, LH, LW, LC))
        z = jnp.concatenate([SOS, z], axis=1)

        xz, yz = z[:, :-1], z[:, 1:]

        ytz, kl_loss = self.decoder(xz, mask, deterministic)

        ytz = self.end(ytz)

        return (yz, ytz), kl_loss

    def vq_call(self, x):
        return self.vq_model(x, True)[0]

    def sample(self, x, deterministic):
        T, H, W, C = x.shape
        z = self.vq_model.encode_vt(x)
        _, LH, LW, LC = z.shape

        SOS = jr.normal(self.make_rng("normal"), (1, 1, LH, LW, LC))
        z = jnp.concat([SOS, jnp.expand_dims(z, 0)], axis=1)[:, :-1]

        for i in tqdm(range(MAX_TEMPORAL_UNITS - T)):
            z_tn = self.end(self.decoder(z, jnp.ones((1, T)), deterministic=deterministic)[0])[:, -1:]
            z = jnp.concat([z, z_tn], axis=1)
            T += 1

        y = self.vq_model.decode_vt(z.reshape(T, LH, LW, LC))
        y = nn.sigmoid(y)
        return y


# Loss function (Adjust KL_WEIGHT according to the use case)
def loss_fn(params, apply, x, det, prng_key, step):
    x, mask, t = x

    (y_true, y_pred), kl_loss = apply(params, x, mask=mask, deterministic=det, rngs={'normal': prng_key})

    recons = jnp.square(y_true - y_pred).mean(axis=(2, 3, 4)) * mask
    recons = recons.sum() / mask.sum()

    KL_WEIGHT = 1e-6
    kl_loss = kl_loss * KL_WEIGHT

    loss = recons + kl_loss

    loss_metric_values_dict = {
        'lt': {'recons': recons, 'kl': kl_loss},
        'mt': {}
    }
    return loss, loss_metric_values_dict
