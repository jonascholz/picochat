from jax import numpy as jnp
from flax import nnx


class GPT2(nnx.Module):
    def __init__(self, vocab_size, D_HIDDEN, rngs):
        self.embed = nnx.Embed(num_embeddings=vocab_size, features=D_HIDDEN, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=4,
            in_features=D_HIDDEN,
            qkv_features=D_HIDDEN,
            decode=False,
            rngs=rngs,
        )

    def __call__(self, x):
        embedded = self.embed(x)
        attended = self.attention(embedded, embedded, embedded)

        return attended
