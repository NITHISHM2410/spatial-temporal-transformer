import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


def attention(q, k, v, d, mask):
    perm = (0, 1, 3, 2)

    qk_w = (q @ k.transpose(*perm)) / jnp.sqrt(d)

    if mask is not False:
        qk_w += mask

    qk_w = nn.softmax(qk_w, -1)
    y = qk_w @ v
    return y


class UpSample(nn.Module):
    conv_out: int

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method='nearest')
        x = nn.Conv(self.conv_out, (3, 3), padding="SAME")(x)
        return x


class DownSample(nn.Module):
    conv_out: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.conv_out, (3, 3), (2, 2), padding="SAME")(x)
        return x


class ResBlock(nn.Module):
    c_in: int
    c_out: int
    norm_g: int
    dim: int = 2

    @nn.compact
    def __call__(self, h, **kwargs):
        hr = h

        h = nn.swish(nn.GroupNorm(self.norm_g)(h))
        h = nn.Conv(self.c_out, (3,) * self.dim, padding='SAME')(h)

        h = nn.swish(nn.GroupNorm(self.norm_g)(h))
        h = nn.Conv(self.c_out, (3,) * self.dim, padding='SAME')(h)

        if self.c_in != self.c_out:
            hr = nn.Conv(self.c_out, (1,) * self.dim)(hr)

        return h + hr


class SpatialSelfAttention(nn.Module):
    heads: int
    norm_g: int

    @nn.compact
    def __call__(self, h):
        h = nn.GroupNorm(self.norm_g)(h)

        B, H, W, C = h.shape
        cph = C // self.heads

        x = nn.Conv(3 * C, (1, 1))(h)
        q, k, v = jnp.split(x.reshape(B, H * W, self.heads, 3 * cph).transpose(0, 2, 1, 3), 3, axis=-1)

        op = attention(q, k, v, cph, mask=False)

        op = op.transpose(0, 2, 1, 3).reshape(B, H, W, C)
        op = nn.Conv(C, (1, 1))(op) + h

        return op


class Encoder(nn.Module):
    c_out: int
    ch_list: Sequence[int]
    attn_res: Sequence[int]
    heads: int
    cph: int
    img_res: int
    norm_g: int

    @nn.compact
    def __call__(self, x):
        num_res = len(self.ch_list)
        cur_res = self.img_res

        x = nn.Conv(self.ch_list[0], (3, 3), padding='same')(x)

        for level in range(num_res):
            block_in = self.ch_list[max(level - 1, 0)]
            block_out = self.ch_list[level]

            for block in range(2):
                x = ResBlock(block_in, block_out, self.norm_g)(x)
                block_in = block_out
                if cur_res in self.attn_res:
                    x = SpatialSelfAttention(block_in // self.cph if self.heads == -1 else self.heads, self.norm_g)(x)

            if level != num_res - 1:
                x = DownSample(self.ch_list[level])(x)
                cur_res //= 2

        x = ResBlock(self.ch_list[-1], self.ch_list[-1], self.norm_g)(x)
        x = SpatialSelfAttention(self.ch_list[-1] // self.cph if self.heads == -1 else self.heads, self.norm_g)(x)
        x = ResBlock(self.ch_list[-1], self.ch_list[-1], self.norm_g)(x)

        x = nn.swish(nn.GroupNorm(self.norm_g)(x))
        x = nn.Conv(self.c_out, (3, 3), padding='SAME')(x)
        return x


class Decoder(nn.Module):
    c_out: int
    ch_list: Sequence[int]
    attn_res: Sequence[int]
    heads: int
    cph: int
    img_res: int
    norm_g: int

    @nn.compact
    def __call__(self, x):
        num_res = len(self.ch_list)
        cur_res = self.img_res

        x = nn.Conv(self.ch_list[-1], (3, 3), padding='SAME')(x)

        x = ResBlock(self.ch_list[-1], self.ch_list[-1], self.norm_g)(x)
        x = SpatialSelfAttention(self.ch_list[-1] // self.cph if self.heads == -1 else self.heads, self.norm_g)(x)
        x = ResBlock(self.ch_list[-1], self.ch_list[-1], self.norm_g)(x)

        for level in reversed(range(num_res)):
            block_in = self.ch_list[min(level + 1, num_res - 1)]
            block_out = self.ch_list[level]

            for block in range(3):
                x = ResBlock(block_in, block_out, self.norm_g)(x)
                block_in = block_out
                if cur_res in self.attn_res:
                    x = SpatialSelfAttention(block_in // self.cph if self.heads == -1 else self.heads, self.norm_g)(x)

            if level != 0:
                x = UpSample(self.ch_list[level])(x)
                cur_res *= 2

        x = nn.swish(nn.GroupNorm(self.norm_g)(x))
        x = nn.Conv(self.c_out, (3, 3), padding='SAME')(x)
        return x


class VQLayer(nn.Module):
    latent_dim: int
    latent_vectors: int

    def setup(self, ):
        self.embedding = self.param('embedding', nn.initializers.normal(), (self.latent_vectors, self.latent_dim))

    @nn.compact
    def __call__(self, inputs):
        z = inputs
        B, H, W, C = z.shape
        z_flatten = z.reshape(B * H * W, self.latent_dim)

        distances = jnp.sum(z_flatten ** 2, axis=-1, keepdims=True) + jnp.sum(self.embedding ** 2,
                                                                              axis=-1) - 2 * jnp.matmul(z_flatten,
                                                                                                        self.embedding.T)
        min_ind = jnp.argmin(distances, axis=-1)
        zq = self.ind2embed(min_ind)
        zq = zq.reshape(B, H, W, self.latent_dim)

        loss = 0.25 * (jnp.square(jax.lax.stop_gradient(zq) - z)).mean() + (
            jnp.square(zq - jax.lax.stop_gradient(z))).mean()
        zq = z + jax.lax.stop_gradient(zq - z)

        return zq, min_ind.reshape(B, H, W), loss

    def ind2embed(self, indices):
        return jnp.take(self.embedding, indices, axis=0)


class VQVAE(nn.Module):
    c_out: int
    zc: int
    ch_list: Sequence[int]
    attn_res: Sequence[int]
    embed_dim: int
    n_embed: int
    heads: int
    cph: int
    img_res: int
    norm_g: int

    def setup(self) -> None:
        self.encoder = Encoder(self.zc, self.ch_list, self.attn_res, self.heads, self.cph, self.img_res, self.norm_g)
        self.decoder = Decoder(self.c_out, self.ch_list, self.attn_res, self.heads, self.cph,
                               self.img_res // (2 ** (len(self.ch_list) - 1)), self.norm_g)
        self.quantizer = VQLayer(self.embed_dim, self.n_embed)
        self.pre_lat = nn.Conv(self.embed_dim, (1, 1))
        self.post_lat = nn.Conv(self.zc, (1, 1))

    def encode(self, x):
        x = self.encoder(x)
        x = self.pre_lat(x)
        zq, min_ind, loss = self.quantizer(x)
        return zq, min_ind, loss

    def decode(self, x):
        x = self.post_lat(x)
        x = self.decoder(x)
        return x

    def encode_vt(self, x):
        x = self.encoder(x)
        x = self.pre_lat(x)
        return x

    def decode_vt(self, x):
        x, _, _ = self.quantizer(x)
        x = self.post_lat(x)
        x = self.decoder(x)
        return x

    @nn.compact
    def __call__(self, x, deterministic):
        zq, min_ind, codebook_loss = self.encode(x)
        y = self.decode(zq)
        return y, codebook_loss


# Loss function's signature is designed w.r.t `flax-pilot` Trainer.

# Change loss_fn below based on use case like from changing BCE to MSE in case of RGB Images,
# or add a new loss function retaining the signature of the function and assign it to trainer during training.

def loss_fn(params, variables, apply, sample, deterministic, prng_key, step, objective):
    x = sample

    y, vq_loss = apply(params, x, deterministic=deterministic, rngs={'normal': prng_key})
    y = nn.sigmoid(y)

    # BCE
    recons_loss = -(x * jnp.log(y + 0.00001) + (1 - x) * jnp.log(1 - y + 0.00001)).mean()

    loss = recons_loss + vq_loss

    loss_metric_values_dict = {
        'lt': {'vq': vq_loss, 'recons': recons_loss},
        'mt': dict()
    }
    return loss, {}, loss_metric_values_dict
