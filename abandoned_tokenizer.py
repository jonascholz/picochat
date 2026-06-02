from jax import numpy as jnp
import string


def main():
    # parse text into long character list
    with open("data.txt", "r") as f:
        text = f.read()
    chars = list(text)

    dict = build_dict(chars)
    print(dict)


def build_dict(chars):
    uniques = list(set(chars))
    n_dict_items = len(uniques) + 30
    output_dict = {}
    # add base characters
    for i in range(len(uniques)):
        output_dict[uniques[i]] = i
    # add paired characters
    for i in range(n_dict_items):
        sorted_pairs = pair_counts(chars)
        most_frequent_pair = sorted_pairs[0][0]
        dict_idx = i + len(uniques)
        output_dict[most_frequent_pair] = dict_idx
        joined = "".join(chars).replace(most_frequent_pair, f"\{dict_idx}")
        chars = list(joined)

    return output_dict


def pair_counts(chars):
    pairs = {}
    for i in range(len(chars) - 1):
        if is_cross_category_merge(chars[i], chars[i + 1]):
            continue
        key = chars[i] + chars[i + 1]
        pairs[key] = pairs[key] + 1 if key in pairs else 1

    sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
    return sorted_pairs


def is_cross_category_merge(a, b):
    category_a = get_category(a)
    category_b = get_category(b)
    return category_a != category_b


def get_category(x):
    category = "letter"
    if x.isdigit():
        category = "digit"
    elif x in string.punctuation:
        category = "punctuation"
    elif x.isspace():
        category = "space"
    return category


if __name__ == "__main__":
    main()
