import logging

import nltk
import spacy

from .config import MAX_SPACY_CHARS

logger = logging.getLogger(__name__)

_NLTK_PACKAGES = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'wordnet': 'corpora/wordnet',
    'vader_lexicon': 'sentiment/vader_lexicon.zip',
}


def download_nltk_data() -> None:
    for pkg, path in _NLTK_PACKAGES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass


def load_spacy_model():
    try:
        model = spacy.load('en_core_web_sm')
    except OSError:
        logger.error("spaCy 'en_core_web_sm' model not found.")
        logger.error("Please download it by running: python -m spacy download en_core_web_sm")
        return None
    model.max_length = max(model.max_length, MAX_SPACY_CHARS)
    return model


download_nltk_data()
nlp = load_spacy_model()
