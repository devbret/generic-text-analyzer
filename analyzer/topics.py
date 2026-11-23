import re

from gensim import corpora
from gensim.models import LdaModel

from .config import NUM_TOPICS
from .preprocessing import preprocess


def build_lda_documents(text: str, sentences: list):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) >= 5:
        chunks = paragraphs
    else:
        num_chunks = min(20, max(5, len(sentences) // 5))
        chunk_size = max(1, len(sentences) // num_chunks)
        chunks = [
            " ".join(sentences[i:i + chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]

    documents = [preprocess(chunk) for chunk in chunks]
    return [doc for doc in documents if doc]


def discover_topics(text: str, sentences: list):
    lda_documents = build_lda_documents(text, sentences)
    if len(lda_documents) < 3:
        return 0, ["Topic modeling skipped: the document could not be split into enough sections."]

    try:
        num_topics = min(NUM_TOPICS, max(2, len(lda_documents) // 3))
        dictionary = corpora.Dictionary(lda_documents)
        if len(dictionary) > 50:
            dictionary.filter_extremes(no_below=2, no_above=0.5)
        corpus = [bow for bow in (dictionary.doc2bow(doc) for doc in lda_documents) if bow]
        if not corpus:
            raise ValueError("vocabulary is empty after filtering")
        num_topics = min(num_topics, len(corpus))
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
        return num_topics, lda_model.print_topics(num_words=5)
    except Exception as e:
        return 0, [f"Could not perform topic modeling: {e}"]
