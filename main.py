from jax import numpy as jnp
import tiktoken


def main():
    with open("data.txt", "r") as f:
        text = f.read()
    encoding = tiktoken.get_encoding("cl100k_base")
    data_as_tokens = encoding.encode(text)
    print(data_as_tokens)


if __name__ == "__main__":
    main()
