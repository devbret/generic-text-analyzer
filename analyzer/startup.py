import importlib
import logging
import sys

logger = logging.getLogger(__name__)

_NOISY_LOGGERS = ("gensim", "httpx", "httpcore", "anthropic")

_REQUIRED_LIBRARIES = [
    (("nltk",), ["NLTK library not found. Please run: pip install nltk"]),
    (("textblob",), ["TextBlob library not found. Please run: pip install textblob"]),
    (("spacy",), [
        "spaCy library not found. Please run: pip install spacy",
        "You also need to download a spaCy model, e.g., python -m spacy download en_core_web_sm",
    ]),
    (("textstat",), ["textstat library not found. Please run: pip install textstat"]),
    (("wordcloud", "matplotlib", "seaborn", "gensim", "sklearn.feature_extraction.text"), [
        "A required library is missing: {missing}. Please install it.",
        "Required: 'wordcloud', 'matplotlib', 'seaborn', 'gensim', 'scikit-learn'",
    ]),
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def verify_dependencies() -> None:
    for modules, messages in _REQUIRED_LIBRARIES:
        for module in modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                for message in messages:
                    logger.error(message.format(missing=e.name))
                sys.exit()
