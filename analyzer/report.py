import logging
import os
from collections import Counter
from dataclasses import dataclass

from .config import OUTPUT_DIR, TOP_N_ENTITIES, TOP_N_NGRAMS, TOP_N_POS, TOP_N_WORDS

logger = logging.getLogger(__name__)

_RULE = "=" * 60


@dataclass
class AnalysisResult:
    file_path: str
    word_count: int
    sentence_count: int
    average_word_length: float
    readability: dict
    sentiment: object
    sentiment_arc: list
    word_freq: Counter
    bigram_freq: Counter
    trigram_freq: Counter
    quadgram_freq: Counter
    fivegram_freq: Counter
    noun_freq: Counter
    verb_freq: Counter
    adjective_freq: Counter
    named_entities: Counter
    entity_profiles: dict
    tfidf_word_scores: list
    key_concepts: set
    entity_concepts: set
    num_topics: int
    topics: list


def _format_counts(pairs) -> str:
    return ", ".join(f"{item} ({count})" for item, count in pairs)


def build_report(result: AnalysisResult) -> str:
    report_lines = []

    report_lines.append(_RULE)
    report_lines.append("AUTOMATED SENSE-MAKING INSIGHTS")
    report_lines.append(_RULE)

    report_lines.append("\n--- Key Thematic Concepts (Noun & TF-IDF Overlap) ---")
    if result.key_concepts:
        report_lines.append(", ".join(sorted(result.key_concepts)))
    else:
        report_lines.append("No significant overlap found between top nouns and TF-IDF keywords.")

    report_lines.append("\n--- Core Entities (Key Themes that are also Entities) ---")
    if result.entity_concepts:
        report_lines.append(", ".join(sorted(result.entity_concepts)))
    else:
        report_lines.append("No key concepts were also identified as top named entities.")
    report_lines.append("\n" + _RULE)

    report_lines.append(f"\nEnhanced Text Analysis Report for: {result.file_path}")
    report_lines.append(_RULE)

    report_lines.append("\n--- Basic Statistics ---")
    report_lines.append(f"Total Word Count (raw): {result.word_count}")
    report_lines.append(f"Total Sentence Count: {result.sentence_count}")
    report_lines.append(f"Unique Lemmatized Words (filtered): {len(result.word_freq)}")
    report_lines.append(f"Average Word Length: {result.average_word_length:.2f}")

    scores = result.readability
    report_lines.append("\n--- Readability Scores ---")
    report_lines.append(f"Flesch Reading Ease: {scores['flesch_reading_ease']:.2f} (Higher is easier)")
    report_lines.append(f"Flesch-Kincaid Grade Level: {scores['flesch_kincaid_grade']}")
    report_lines.append(f"Gunning Fog Index: {scores['gunning_fog']:.2f}")
    report_lines.append(f"SMOG Index: {scores['smog_index']}")

    report_lines.append("\n--- Sentiment Analysis (Overall) ---")
    report_lines.append(f"Polarity: {result.sentiment.polarity:.2f} (Ranges from -1 [negative] to +1 [positive])")
    report_lines.append(f"Subjectivity: {result.sentiment.subjectivity:.2f} (Ranges from 0 [objective] to 1 [subjective])")

    report_lines.append(f"\n--- Sentiment Arc (document split into {len(result.sentiment_arc)} parts, VADER compound) ---")
    report_lines.append(" -> ".join(f"{score:.2f}" for score in result.sentiment_arc))

    report_lines.append(f"\n--- Discovered Topics (LDA, {result.num_topics} topics) ---")
    for topic in result.topics:
        if isinstance(topic, tuple) and len(topic) == 2:
            topic_num, topic_words = topic
            report_lines.append(f"Topic {topic_num + 1}: {topic_words}")
        else:
            report_lines.append(str(topic))

    report_lines.append(f"\n--- Top {TOP_N_WORDS} Keywords (by TF-IDF score) ---")
    for word, score in result.tfidf_word_scores[:TOP_N_WORDS]:
        report_lines.append(f"{word}: {score:.4f}")

    if result.named_entities:
        report_lines.append(f"\n--- Top {TOP_N_ENTITIES} Named Entities ---")
        for (entity, label), count in result.named_entities.most_common(TOP_N_ENTITIES):
            report_lines.append(f"{entity} ({label}): {count}")

    if result.entity_profiles:
        report_lines.append("\n--- Key Entity Profiles ---")
        for entity, profile in result.entity_profiles.items():
            report_lines.append(f"\nProfile for '{entity}':")
            if profile['actions']:
                report_lines.append(f"  - Common Actions: {_format_counts(profile['actions'])}")
            if profile['descriptors']:
                report_lines.append(f"  - Common Descriptors: {_format_counts(profile['descriptors'])}")

    pos_sections = [
        ("Nouns", result.noun_freq),
        ("Verbs", result.verb_freq),
        ("Adjectives", result.adjective_freq),
    ]
    for label, freq in pos_sections:
        report_lines.append(f"\n--- Top {TOP_N_POS} Most Common {label} (Lemmatized) ---")
        for word, count in freq.most_common(TOP_N_POS):
            report_lines.append(f"{word}: {count}")

    ngram_sections = [
        ("Bigrams", result.bigram_freq),
        ("Trigrams", result.trigram_freq),
        ("Quadgrams", result.quadgram_freq),
        ("Fivegrams", result.fivegram_freq),
    ]
    for label, freq in ngram_sections:
        report_lines.append(f"\n--- Top {TOP_N_NGRAMS} Most Common {label} (Lemmatized) ---")
        for gram, count in freq.most_common(TOP_N_NGRAMS):
            report_lines.append(f"{' '.join(gram)}: {count}")

    return "\n".join(report_lines)


def save_report(report: str, output_base: str) -> str:
    output_filepath = os.path.join(OUTPUT_DIR, f"{output_base}_enhanced_analysis_report.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info("Report saved to: %s", output_filepath)
    except Exception as e:
        logger.error("Could not save report: %s", e)

    return output_filepath
