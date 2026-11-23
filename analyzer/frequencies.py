from collections import Counter

from nltk.util import ngrams


def ngram_frequencies(tokens, n: int) -> Counter:
    return Counter(ngrams(tokens, n))


def pos_frequencies(tagged_tokens, coarse_tag: str) -> Counter:
    return Counter(word for word, coarse in tagged_tokens if coarse == coarse_tag)
