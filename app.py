import sys
import re
import os
import logging
import anthropic
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers (gensim's per-pass training trace, the
# anthropic/httpx request log) so only this script's progress is shown.
for _noisy in ("gensim", "httpx", "httpcore", "anthropic"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.util import ngrams
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag
except ImportError:
    logger.error("NLTK library not found. Please run: pip install nltk")
    sys.exit()

try:
    from textblob import TextBlob
except ImportError:
    logger.error("TextBlob library not found. Please run: pip install textblob")
    sys.exit()

try:
    import spacy
except ImportError:
    logger.error("spaCy library not found. Please run: pip install spacy")
    logger.error("You also need to download a spaCy model, e.g., python -m spacy download en_core_web_sm")
    sys.exit()

try:
    import textstat
except ImportError:
    logger.error("textstat library not found. Please run: pip install textstat")
    sys.exit()

try:
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use("Agg")  # headless/multiprocess-safe backend for chart generation
    import matplotlib.pyplot as plt
    import seaborn as sns
    from gensim import corpora
    from gensim.models import LdaModel
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as e:
    logger.error("A required library is missing: %s. Please install it.", e.name)
    logger.error("Required: 'wordcloud', 'matplotlib', 'seaborn', 'gensim', 'scikit-learn'")
    sys.exit()

CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-20250514")
TOP_N_WORDS = 50
TOP_N_NGRAMS = 50
TOP_N_ENTITIES = 50
TOP_N_POS = 50
NUM_TOPICS = 50
SUMMARY_SENTENCE_COUNT = 43
MAX_SPACY_CHARS = 250_000

def download_nltk_data():
    needed = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'wordnet': 'corpora/wordnet',
        'vader_lexicon': 'sentiment/vader_lexicon.zip',
    }
    for pkg, path in needed.items():
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

download_nltk_data()

try:
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = max(nlp.max_length, MAX_SPACY_CHARS)
except OSError:
    logger.error("spaCy 'en_core_web_sm' model not found.")
    logger.error("Please download it by running: python -m spacy download en_core_web_sm")
    nlp = None


def summarize_txt_file_with_claude(filepath: str) -> str | None:
    logger.info("Reading and summarizing: %s", filepath)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except Exception as e:
        logger.error("Error reading file %s: %s", filepath, e)
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Anthropic API key not found. Please check your .env file.")
        return None

    # The Anthropic SDK automatically retries 429/5xx/529 (overloaded) responses
    # with exponential backoff, so no manual retry loop is needed.
    client = anthropic.Anthropic(api_key=api_key, max_retries=5)

    try:
        logger.info("Connecting to Anthropic API for text summary...")
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4300,
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Please summarize the following document in clear and concise language suitable for a general audience.

                    Highlight:
                    - The main topics covered
                    - Any key arguments or insights
                    - Notable conclusions, themes, or recommendations

                    Here is the text:
                    {file_content}
                    """
                }
            ],
        )
        logger.info("Summary successfully generated.")
        return response.content[0].text
    except Exception as e:
        logger.error("Anthropic API error: %s", e)
        logger.error("Please check your .env file and network connection.")
        return None

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

def _preprocess_nltk(text: str):
    text_lower = text.lower()
    text_cleaned = re.sub(r"[^\w\s]", " ", text_lower)

    tokens = word_tokenize(text_cleaned)

    stop_words = set(stopwords.words('english'))
    custom_stopwords = ["uh", "um", "like", "yeah", "im", "dont", "go", "know", "going", "thats", "think", "let", "lets"]
    stop_words.update(custom_stopwords)
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    lemmatizer = WordNetLemmatizer()
    pos_tags_for_lemma = pos_tag(filtered_tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in pos_tags_for_lemma]

    return lemmatized_tokens

def _preprocess(text: str):
    text = re.sub(r"[^\w\s]", " ", text.lower())
    
    if nlp:
        doc = nlp(text)
        stop_words = set(stopwords.words('english'))
        custom = {"uh","um","like","yeah","im","dont","go","know","going","thats","think","let","lets"}
        stop_words |= custom
        lemmatized = [
            t.lemma_.lower()
            for t in doc
            if (
                not t.is_space
                and not t.like_num
                and t.lemma_.isalpha()
                and t.text.lower() not in stop_words
                and not t.is_stop
                and len(t) > 1
            )
        ]
        return lemmatized
    else:
        return _preprocess_nltk(text)

def _build_lda_documents(text: str, sentences: list):
    """Split a single document into multiple pseudo-documents for topic modeling.

    LDA learns topics from word co-occurrence *across* documents, so running it on
    one bag-of-words is meaningless. Prefer paragraph boundaries; fall back to
    grouping sentences into chunks when there are too few paragraphs.
    """
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

    documents = [_preprocess(chunk) for chunk in chunks]
    return [doc for doc in documents if doc]

def _generate_visualizations(output_base, word_freq, bigram_freq, named_entities, sentiment_arc):
    logger.info("Generating visualizations...")

    try:
        wordcloud = WordCloud(width=1200, height=600, background_color='white', colormap='viridis', collocations=False).generate_from_frequencies(word_freq)
        wordcloud_filename = f"{output_base}_wordcloud.png"
        os.makedirs("output", exist_ok=True)
        wordcloud.to_file(f"output/{wordcloud_filename}")
        logger.info("Word cloud saved to: %s", wordcloud_filename)
    except Exception as e:
        logger.error("Could not generate word cloud: %s", e)

    plt.style.use('seaborn-v0_8-whitegrid')

    def create_bar_chart(data, title, filename):
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
            os.makedirs("output", exist_ok=True)
            plt.savefig(f"output/{filename}")
            plt.close()
            logger.info("Chart saved to: %s", filename)
        except Exception as e:
            logger.error("Could not generate chart '%s': %s", title, e)

    def create_sentiment_arc_chart(data, filename):
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
            os.makedirs("output", exist_ok=True)
            plt.savefig(f"output/{filename}")
            plt.close()
            logger.info("Chart saved to: %s", filename)
        except Exception as e:
            logger.error("Could not generate sentiment arc chart: %s", e)

    create_bar_chart(word_freq.most_common(TOP_N_WORDS),
                     f'Top {TOP_N_WORDS} Most Common Words (Lemmatized)',
                     f"{output_base}_top_words_chart.png")

    top_bigrams = bigram_freq.most_common(TOP_N_NGRAMS)
    bigram_labels = [' '.join(b) for b, c in top_bigrams]
    bigram_counts = [c for b, c in top_bigrams]
    create_bar_chart(list(zip(bigram_labels, bigram_counts)),
                     f'Top {TOP_N_NGRAMS} Most Common Bigrams',
                     f"{output_base}_top_bigrams_chart.png")

    if named_entities:
        top_entities = named_entities.most_common(TOP_N_ENTITIES)
        entity_labels = [f"{ent[0]} ({ent[1]})" for ent, count in top_entities]
        entity_counts = [count for ent, count in top_entities]
        create_bar_chart(list(zip(entity_labels, entity_counts)),
                         f'Top {TOP_N_ENTITIES} Named Entities',
                         f"{output_base}_top_entities_chart.png")

    if sentiment_arc:
        create_sentiment_arc_chart(sentiment_arc, f"{output_base}_sentiment_arc.png")

def analyze_text(file_path):
    logger.info("Starting analysis for: %s", file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        logger.error("The file at '%s' was not found.", file_path)
        return
    except Exception as e:
        logger.error("An error occurred while reading the file: %s", e)
        return

    if not text.strip():
        logger.warning("The file is empty. Analysis cannot proceed.")
        return

    lemmatized_tokens = _preprocess(text)
    if not lemmatized_tokens:
        logger.warning("The text is too short or contains only stopwords. Analysis cannot proceed.")
        return
    
    sentences = sent_tokenize(text)
    original_tokens = word_tokenize(text.lower())

    word_freq = Counter(lemmatized_tokens)
    bigram_freq = Counter(ngrams(lemmatized_tokens, 2))
    trigram_freq = Counter(ngrams(lemmatized_tokens, 3))
    quadgram_freq = Counter(ngrams(lemmatized_tokens, 4))
    fivegram_freq = Counter(ngrams(lemmatized_tokens, 5))

    sentiment = TextBlob(text).sentiment

    pos_tags = pos_tag(lemmatized_tokens)
    nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
    verbs = [word for word, tag in pos_tags if tag.startswith('VB')]
    adjectives = [word for word, tag in pos_tags if tag.startswith('JJ')]
    noun_freq = Counter(nouns)
    verb_freq = Counter(verbs)
    adjective_freq = Counter(adjectives)

    all_ents = Counter()
    if nlp:
        for chunk in split_text(text, MAX_SPACY_CHARS):
            doc_chunk = nlp(chunk)
            all_ents.update([(ent.text.strip(), ent.label_) for ent in doc_chunk.ents])
    named_entities = all_ents

    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=word_tokenize, token_pattern=None)
    try:
        tfidf_matrix = vectorizer.fit_transform(sent_tokenize(text))
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).tolist()[0]
        tfidf_word_scores = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    except ValueError:
        tfidf_word_scores = []

    lda_documents = _build_lda_documents(text, sentences)
    if len(lda_documents) < 3:
        num_topics = 0
        topics = ["Topic modeling skipped: the document could not be split into enough sections."]
    else:
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
            topics = lda_model.print_topics(num_words=5)
        except Exception as e:
            num_topics = 0
            topics = [f"Could not perform topic modeling: {e}"]
    
    top_nouns_set = {word for word, count in noun_freq.most_common(TOP_N_WORDS)}
    top_tfidf_set = {word for word, score in tfidf_word_scores[:TOP_N_WORDS]}
    top_entity_text_set = {ent[0].lower() for ent, count in named_entities.most_common(TOP_N_ENTITIES)}
    key_concepts = top_nouns_set.intersection(top_tfidf_set)
    entity_concepts = key_concepts.intersection(top_entity_text_set)

    entity_profiles = {} 
    if nlp and named_entities:
        logger.info("Starting entity profiling analysis (processing text in chunks)...")
        top_entities_to_profile = [ent[0] for ent, count in named_entities.most_common(10)]
        
        aggregated_profiles = {
            entity_text: {'actions': Counter(), 'descriptors': Counter()}
            for entity_text in top_entities_to_profile
        }

        for chunk in split_text(text, MAX_SPACY_CHARS):
            doc_chunk = nlp(chunk)
            for ent in doc_chunk.ents:
                if ent.text in aggregated_profiles:
                    sent = ent.sent
                    for token in sent:
                        if token.dep_ in ('nsubj', 'nsubjpass') and ent.start <= token.i <= ent.end - 1:
                            if token.head.pos_ == 'VERB':
                                aggregated_profiles[ent.text]['actions'][token.head.lemma_] += 1
                        if token.dep_ == 'amod' and ent.start <= token.head.i <= ent.end - 1:
                            aggregated_profiles[ent.text]['descriptors'][token.lemma_] += 1
        
        for entity, counters in aggregated_profiles.items():
            actions = counters['actions'].most_common(5)
            descriptors = counters['descriptors'].most_common(5)
            if actions or descriptors:
                entity_profiles[entity] = {'actions': actions, 'descriptors': descriptors}

    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

    sia = SentimentIntensityAnalyzer()
    sentiment_arc = []
    if sentences:
        num_chunks = 20
        chunk_size = max(1, len(sentences) // num_chunks)
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_chunks - 1 else len(sentences)
            if start < len(sentences):
                chunk_text = " ".join(sentences[start:end])
                if chunk_text:
                    compound = sia.polarity_scores(chunk_text)['compound']
                    sentiment_arc.append(compound)

    report_lines = []
    
    report_lines.append("=" * 60)
    report_lines.append("AUTOMATED SENSE-MAKING INSIGHTS")
    report_lines.append("=" * 60)
    
    report_lines.append("\n--- Key Thematic Concepts (Noun & TF-IDF Overlap) ---")
    if key_concepts:
        report_lines.append(", ".join(sorted(list(key_concepts))))
    else:
        report_lines.append("No significant overlap found between top nouns and TF-IDF keywords.")

    report_lines.append("\n--- Core Entities (Key Themes that are also Entities) ---")
    if entity_concepts:
        report_lines.append(", ".join(sorted(list(entity_concepts))))
    else:
        report_lines.append("No key concepts were also identified as top named entities.")
    report_lines.append("\n" + "=" * 60)

    report_lines.append(f"\nEnhanced Text Analysis Report for: {file_path}")
    report_lines.append("=" * 60)

    report_lines.append("\n--- Basic Statistics ---")
    report_lines.append(f"Total Word Count (raw): {len(original_tokens)}")
    report_lines.append(f"Total Sentence Count: {len(sentences)}")
    report_lines.append(f"Unique Lemmatized Words (filtered): {len(word_freq)}")
    alpha_tokens = [t for t in original_tokens if t.isalpha()]
    avg_len = (sum(map(len, alpha_tokens)) / len(alpha_tokens)) if alpha_tokens else 0
    report_lines.append(f"Average Word Length: {avg_len:.2f}")

    report_lines.append("\n--- Readability Scores ---")
    report_lines.append(f"Flesch Reading Ease: {textstat.flesch_reading_ease(text):.2f} (Higher is easier)")
    report_lines.append(f"Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(text)}")
    report_lines.append(f"Gunning Fog Index: {textstat.gunning_fog(text):.2f}")
    report_lines.append(f"SMOG Index: {textstat.smog_index(text)}")

    report_lines.append("\n--- Sentiment Analysis (Overall) ---")
    report_lines.append(f"Polarity: {sentiment.polarity:.2f} (Ranges from -1 [negative] to +1 [positive])")
    report_lines.append(f"Subjectivity: {sentiment.subjectivity:.2f} (Ranges from 0 [objective] to 1 [subjective])")
    
    report_lines.append(f"\n--- Sentiment Arc (document split into {len(sentiment_arc)} parts, VADER compound) ---")
    arc_scores_str = [f"{score:.2f}" for score in sentiment_arc]
    report_lines.append(" -> ".join(arc_scores_str))

    report_lines.append(f"\n--- Discovered Topics (LDA, {num_topics} topics) ---")
    for topic in topics:
        if isinstance(topic, tuple) and len(topic) == 2:
            topic_num, topic_words = topic
            report_lines.append(f"Topic {topic_num + 1}: {topic_words}")
        else:
            report_lines.append(str(topic))

    report_lines.append(f"\n--- Top {TOP_N_WORDS} Keywords (by TF-IDF score) ---")
    for word, score in tfidf_word_scores[:TOP_N_WORDS]:
        report_lines.append(f"{word}: {score:.4f}")

    if named_entities:
        report_lines.append(f"\n--- Top {TOP_N_ENTITIES} Named Entities ---")
        for (entity, label), count in named_entities.most_common(TOP_N_ENTITIES):
            report_lines.append(f"{entity} ({label}): {count}")
    
    if entity_profiles:
        report_lines.append(f"\n--- Key Entity Profiles ---")
        for entity, profile in entity_profiles.items():
            report_lines.append(f"\nProfile for '{entity}':")
            if profile['actions']:
                action_str = ", ".join([f"{v[0]} ({v[1]})" for v in profile['actions']])
                report_lines.append(f"  - Common Actions: {action_str}")
            if profile['descriptors']:
                desc_str = ", ".join([f"{d[0]} ({d[1]})" for d in profile['descriptors']])
                report_lines.append(f"  - Common Descriptors: {desc_str}")

    report_lines.append(f"\n--- Top {TOP_N_POS} Most Common Nouns (Lemmatized) ---")
    for word, count in noun_freq.most_common(TOP_N_POS):
        report_lines.append(f"{word}: {count}")

    report_lines.append(f"\n--- Top {TOP_N_POS} Most Common Verbs (Lemmatized) ---")
    for word, count in verb_freq.most_common(TOP_N_POS):
        report_lines.append(f"{word}: {count}")

    report_lines.append(f"\n--- Top {TOP_N_POS} Most Common Adjectives (Lemmatized) ---")
    for word, count in adjective_freq.most_common(TOP_N_POS):
        report_lines.append(f"{word}: {count}")
    
    report_lines.append(f"\n--- Top {TOP_N_NGRAMS} Most Common Bigrams (Lemmatized) ---")
    for bigram, count in bigram_freq.most_common(TOP_N_NGRAMS):
        report_lines.append(f"{' '.join(bigram)}: {count}")

    report_lines.append(f"\n--- Top {TOP_N_NGRAMS} Most Common Trigrams (Lemmatized) ---")
    for trigram, count in trigram_freq.most_common(TOP_N_NGRAMS):
        report_lines.append(f"{' '.join(trigram)}: {count}")

    report_lines.append(f"\n--- Top {TOP_N_NGRAMS} Most Common Quadgrams (Lemmatized) ---")
    for quadgram, count in quadgram_freq.most_common(TOP_N_NGRAMS):
        report_lines.append(f"{' '.join(quadgram)}: {count}")

    report_lines.append(f"\n--- Top {TOP_N_NGRAMS} Most Common Fivegrams (Lemmatized) ---")
    for fivegram, count in fivegram_freq.most_common(TOP_N_NGRAMS):
        report_lines.append(f"{' '.join(fivegram)}: {count}")

    final_report = "\n".join(report_lines)

    filename = os.path.basename(file_path)
    basename, _ = os.path.splitext(filename)
    output_filename = f"{basename}_enhanced_analysis_report.txt"
    output_dir = "output"
    output_filepath = os.path.join(output_dir, output_filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(final_report)
        logger.info("Report saved to: %s", output_filepath)
    except Exception as e:
        logger.error("Could not save report: %s", e)

    _generate_visualizations(basename, word_freq, bigram_freq, named_entities, sentiment_arc)
    logger.info("Analysis complete for: %s", file_path)

    summary = summarize_txt_file_with_claude(output_filepath)
    if summary:
        summary_filename = os.path.splitext(output_filepath)[0] + "_claude_summary.md"
        try:
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(summary)
            logger.info("Claude summary saved to: %s", summary_filename)
        except Exception as e:
            logger.error("Could not save Claude summary: %s", e)
    else:
        logger.warning("No summary generated.")

    return {
        "file": file_path,
        "report": output_filepath
    }

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed

    input_dir = "input"
    if not os.path.isdir(input_dir):
        logger.error("Input directory '%s' not found.", input_dir)
        sys.exit(1)

    txt_files = [os.path.join(input_dir, fn) for fn in os.listdir(input_dir) if fn.endswith(".txt")]

    if not txt_files:
        logger.warning("No .txt files found in '%s'.", input_dir)
        sys.exit(0)

    results = []
    max_workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyze_text, path) for path in txt_files]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                logger.error("Error processing a file in parallel: %s", e)
