import logging
import os
from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize

from .entities import build_entity_profiles, extract_named_entities
from .frequencies import ngram_frequencies, pos_frequencies
from .keywords import compute_tfidf_scores, find_key_concepts
from .preprocessing import analyze_tokens
from .report import AnalysisResult, build_report, save_report
from .sentiment import compute_sentiment_arc, overall_sentiment
from .summarizer import save_summary, summarize_txt_file_with_claude
from .text_stats import average_word_length, readability_scores
from .topics import discover_topics
from .visualizations import generate_visualizations

logger = logging.getLogger(__name__)


def _read_text(file_path: str) -> str | None:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logger.error("The file at '%s' was not found.", file_path)
    except Exception as e:
        logger.error("An error occurred while reading the file: %s", e)
    return None


def analyze_text(file_path: str) -> dict | None:
    logger.info("Starting analysis for: %s", file_path)
    text = _read_text(file_path)
    if text is None:
        return None

    if not text.strip():
        logger.warning("The file is empty. Analysis cannot proceed.")
        return None

    lemmatized_tokens, pos_tagged_tokens = analyze_tokens(text)
    if not lemmatized_tokens:
        logger.warning("The text is too short or contains only stopwords. Analysis cannot proceed.")
        return None

    sentences = sent_tokenize(text)
    original_tokens = word_tokenize(text.lower())

    noun_freq = pos_frequencies(pos_tagged_tokens, "NOUN")
    named_entities = extract_named_entities(text)
    tfidf_word_scores = compute_tfidf_scores(sentences)
    key_concepts, entity_concepts = find_key_concepts(noun_freq, tfidf_word_scores, named_entities)
    num_topics, topics = discover_topics(text, sentences)

    result = AnalysisResult(
        file_path=file_path,
        word_count=len(original_tokens),
        sentence_count=len(sentences),
        average_word_length=average_word_length(original_tokens),
        readability=readability_scores(text),
        sentiment=overall_sentiment(text),
        sentiment_arc=compute_sentiment_arc(sentences),
        word_freq=Counter(lemmatized_tokens),
        bigram_freq=ngram_frequencies(lemmatized_tokens, 2),
        trigram_freq=ngram_frequencies(lemmatized_tokens, 3),
        quadgram_freq=ngram_frequencies(lemmatized_tokens, 4),
        fivegram_freq=ngram_frequencies(lemmatized_tokens, 5),
        noun_freq=noun_freq,
        verb_freq=pos_frequencies(pos_tagged_tokens, "VERB"),
        adjective_freq=pos_frequencies(pos_tagged_tokens, "ADJ"),
        named_entities=named_entities,
        entity_profiles=build_entity_profiles(text, named_entities),
        tfidf_word_scores=tfidf_word_scores,
        key_concepts=key_concepts,
        entity_concepts=entity_concepts,
        num_topics=num_topics,
        topics=topics,
    )

    basename = os.path.splitext(os.path.basename(file_path))[0]
    report_path = save_report(build_report(result), basename)

    generate_visualizations(basename, result.word_freq, result.bigram_freq,
                            result.named_entities, result.sentiment_arc)
    logger.info("Analysis complete for: %s", file_path)

    summary = summarize_txt_file_with_claude(report_path)
    if summary:
        save_summary(summary, report_path)
    else:
        logger.warning("No summary generated.")

    return {
        "file": file_path,
        "report": report_path
    }
