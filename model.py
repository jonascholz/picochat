from jax import numpy as jnp
from flax import nnx

N_HEADS = 4


class GPT2(nnx.Module):

    def __init__(self, vocab_size, D_HIDDEN, rngs):
        self.embed = nnx.Embed(num_embeddings=vocab_size, features=D_HIDDEN, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=N_HEADS,
            in_features=D_HIDDEN,
            qkv_features=D_HIDDEN,
            decode=False,
            rngs=rngs,
        )
        self.norm = nnx.RMSNorm(num_features=D_HIDDEN, use_scale=False, rngs=None)

    def __call__(self, x):
        x = self.embed(x)
        x = self.attention(x, x, x)
        x = self.norm(x)

        return x
