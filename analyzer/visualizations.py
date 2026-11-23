import logging
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from .config import OUTPUT_DIR, TOP_N_ENTITIES, TOP_N_NGRAMS, TOP_N_WORDS

logger = logging.getLogger(__name__)


def _save_word_cloud(word_freq, filename):
    try:
        wordcloud = WordCloud(width=1200, height=600, background_color='white', colormap='viridis', collocations=False).generate_from_frequencies(word_freq)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        wordcloud.to_file(os.path.join(OUTPUT_DIR, filename))
        logger.info("Word cloud saved to: %s", filename)
    except Exception as e:
        logger.error("Could not generate word cloud: %s", e)


def _save_bar_chart(data, title, filename):
    if not data:
        logger.warning("No data to generate chart: %s", title)
        return
    try:
        items = [item for item, count in data]
        counts = [count for item, count in data]

        plt.figure(figsize=(10, 8))
        sns.barplot(x=counts, y=items, hue=items, legend=False, palette='plasma')
        plt.title(title, fontsize=16)
        plt.xlabel('Frequency', fontsize=12)
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
        logger.info("Chart saved to: %s", filename)
    except Exception as e:
        logger.error("Could not generate chart '%s': %s", title, e)


def _save_sentiment_arc_chart(data, filename):
    if not data:
        logger.warning("No data to generate sentiment arc chart.")
        return
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(data, marker='o', linestyle='-', color='indigo')
        plt.title('Sentiment Arc of the Document', fontsize=16)
        plt.xlabel('Document Progression (Part)', fontsize=12)
        plt.ylabel('Sentiment Polarity', fontsize=12)
        plt.axhline(0, color='grey', linestyle='--')
        plt.xticks(range(len(data)), [str(i+1) for i in range(len(data))])
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
        logger.info("Chart saved to: %s", filename)
    except Exception as e:
        logger.error("Could not generate sentiment arc chart: %s", e)


def generate_visualizations(output_base, word_freq, bigram_freq, named_entities, sentiment_arc):
    logger.info("Generating visualizations...")

    _save_word_cloud(word_freq, f"{output_base}_wordcloud.png")

    plt.style.use('seaborn-v0_8-whitegrid')

    _save_bar_chart(word_freq.most_common(TOP_N_WORDS),
                    f'Top {TOP_N_WORDS} Most Common Words (Lemmatized)',
                    f"{output_base}_top_words_chart.png")

    top_bigrams = [(' '.join(bigram), count) for bigram, count in bigram_freq.most_common(TOP_N_NGRAMS)]
    _save_bar_chart(top_bigrams,
                    f'Top {TOP_N_NGRAMS} Most Common Bigrams',
                    f"{output_base}_top_bigrams_chart.png")

    if named_entities:
        top_entities = [(f"{text} ({label})", count) for (text, label), count in named_entities.most_common(TOP_N_ENTITIES)]
        _save_bar_chart(top_entities,
                        f'Top {TOP_N_ENTITIES} Named Entities',
                        f"{output_base}_top_entities_chart.png")

    if sentiment_arc:
        _save_sentiment_arc_chart(sentiment_arc, f"{output_base}_sentiment_arc.png")
