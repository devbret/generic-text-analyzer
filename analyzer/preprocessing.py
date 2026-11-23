from functools import lru_cache

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from .config import CUSTOM_STOPWORDS, MAX_SPACY_CHARS
from .nlp_resources import nlp

_SPACY_TO_COARSE = {
    "NOUN": "NOUN",
    "PROPN": "NOUN",
    "VERB": "VERB",
    "AUX": "VERB",
    "ADJ": "ADJ",
}


def split_text(text: str, max_len: int = MAX_SPACY_CHARS):
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_len, text_len)
        if end < text_len:
            next_period = text.rfind('.', start, end)
            if next_period != -1 and next_period > start + 0.8 * max_len:
                end = next_period + 1
        yield text[start:end]
        start = end


@lru_cache(maxsize=1)
def _stop_words() -> frozenset:
    words = set(stopwords.words('english'))
    words.update(CUSTOM_STOPWORDS)
    return frozenset(words)


def _get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'


def _ptb_to_coarse(tag: str) -> str | None:
    if tag.startswith("NN"):
        return "NOUN"
    if tag.startswith("VB"):
        return "VERB"
    if tag.startswith("JJ"):
        return "ADJ"
    return None


def _analyze_tokens_spacy(text: str):
    stop_words = _stop_words()
    lemmas = []
    tagged = []

    for chunk in split_text(text, MAX_SPACY_CHARS):
        doc = nlp(chunk)
        for token in doc:
            if token.is_space or token.like_num or token.is_stop:
                continue
            if len(token) <= 1:
                continue
            if token.text.lower() in stop_words:
                continue
            lemma = token.lemma_.lower()
            if not lemma.isalpha():
                continue
            lemmas.append(lemma)
            coarse = _SPACY_TO_COARSE.get(token.pos_)
            if coarse:
                tagged.append((lemma, coarse))

    return lemmas, tagged


def _analyze_tokens_nltk(text: str):
    stop_words = _stop_words()
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    tagged = []

    for sentence in sent_tokenize(text):
        tokens = word_tokenize(sentence)
        for word, tag in pos_tag(tokens):
            lower = word.lower()
            if len(lower) <= 1 or not lower.isalpha() or lower in stop_words:
                continue
            lemma = lemmatizer.lemmatize(lower, _get_wordnet_pos(tag))
            if not lemma.isalpha():
                continue
            lemmas.append(lemma)
            coarse = _ptb_to_coarse(tag)
            if coarse:
                tagged.append((lemma, coarse))

    return lemmas, tagged


def analyze_tokens(text: str):
    if nlp:
        return _analyze_tokens_spacy(text)
    return _analyze_tokens_nltk(text)


def preprocess(text: str):
    return analyze_tokens(text)[0]
