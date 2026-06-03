import tiktoken
import jax
from jax import numpy as jnp
from flax import nnx

from model import GPT2


def main():
    with open("data.txt", "r") as f:
        text = f.read()
    tokenizer = tiktoken.get_encoding("p50k_base")
    data_as_tokens = tokenizer.encode(text)
    first_sample = jnp.array(data_as_tokens[:10])
    second_sample = jnp.array(data_as_tokens[10:20])
    batch_tokens = jnp.stack([first_sample, second_sample])
    D_HIDDEN = 128
    model = GPT2(tokenizer.n_vocab, D_HIDDEN, nnx.Rngs(0))
    attended = model(batch_tokens)
    print(attended.shape)


if __name__ == "__main__":
    main()
