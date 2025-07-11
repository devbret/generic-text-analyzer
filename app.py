import sys
import re
import os
import anthropic
import time
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# --- Library Imports ---

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.util import ngrams
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag
except ImportError:
    print("NLTK library not found. Please run: pip install nltk")
    sys.exit()

try:
    from textblob import TextBlob
except ImportError:
    print("TextBlob library not found. Please run: pip install textblob")
    sys.exit()

try:
    import spacy
except ImportError:
    print("spaCy library not found. Please run: pip install spacy")
    print("You also need to download a spaCy model, e.g., python -m spacy download en_core_web_sm")
    sys.exit()

try:
    import textstat
except ImportError:
    print("textstat library not found. Please run: pip install textstat")
    sys.exit()

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import seaborn as sns
    from gensim import corpora
    from gensim.models import LdaModel
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as e:
    print(f"A required library is missing: {e.name}. Please install it.")
    print("Required: 'wordcloud', 'matplotlib', 'seaborn', 'gensim', 'scikit-learn'")
    sys.exit()


# --- Configuration ---

TOP_N_WORDS = 50
TOP_N_NGRAMS = 50
TOP_N_ENTITIES = 50
TOP_N_POS = 50
NUM_TOPICS = 50
SUMMARY_SENTENCE_COUNT = 10
MAX_SPACY_CHARS = 1_000_000


# --- NLTK and spaCy Data Downloads ---
def download_nltk_data():
    nltk_packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
    for package in nltk_packages:
        try:
            if package == 'punkt':
                 path = f'tokenizers/{package}'
            elif package.endswith('_tagger'):
                 path = f'taggers/{package}'
            else:
                 path = f'corpora/{package}'
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK data '{package}'...")

download_nltk_data()

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("spaCy 'en_core_web_sm' model not found.")
    print("Please download it by running: python -m spacy download en_core_web_sm")
    nlp = None

# --- Helper Functions ---

def summarize_txt_file_with_claude(filepath: str) -> str | None:
    print(f"\nReading and summarizing: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    max_retries = 5
    initial_wait_time = 1

    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        if not client.api_key:
            raise ValueError("Anthropic API key not found.")

        for attempt in range(max_retries):
            try:
                print("\nConnecting to Anthropic API for text summary...")
                print("--- AI-Generated Summary ---")

                response = client.messages.create(
                    model="claude-opus-4-20250514",
                    max_tokens=3000,
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

                print("Summary successfully generated.")
                return response.content[0].text

            except anthropic.APIStatusError as e:
                if e.status_code == 529 and 'overloaded_error' in e.response.text:
                    if attempt < max_retries - 1:
                        wait_time = initial_wait_time * (2 ** attempt)
                        print(f"\nAPI is overloaded. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

    except Exception as e:
        print(f"\n--- ANTHROPIC API ERROR ---")
        print(f"An error occurred: {e}")
        print("Please check your .env file and network connection.")
        print("---------------------------\n")
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
    
def split_text(text, max_len=1_000_000):
    for i in range(0, len(text), max_len):
        yield text[i:i + max_len]

def _preprocess(text):
    text_lower = text.lower()
    text_cleaned = re.sub(r'[^\w\s]', '', text_lower)
    
    tokens = word_tokenize(text_cleaned)
    
    stop_words = set(stopwords.words('english'))
    custom_stopwords = ["uh", "um", "like", "yeah", "im", "dont", "go", "know", "going", "thats", "think", "let", "lets"]
    stop_words.update(custom_stopwords)
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    lemmatizer = WordNetLemmatizer()
    pos_tags_for_lemma = pos_tag(filtered_tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in pos_tags_for_lemma]
    
    return lemmatized_tokens

def _generate_visualizations(output_base, word_freq, bigram_freq, named_entities, sentiment_arc):
    print("\nGenerating visualizations...")
    
    try:
        wordcloud = WordCloud(width=1200, height=600, background_color='white', colormap='viridis').generate_from_frequencies(word_freq)
        wordcloud_filename = f"{output_base}_wordcloud.png"
        wordcloud.to_file(f"output/{wordcloud_filename}")
        print(f"Word cloud saved to: {wordcloud_filename}")
    except Exception as e:
        print(f"Could not generate word cloud: {e}")

    plt.style.use('seaborn-v0_8-whitegrid')

    def create_bar_chart(data, title, filename):
        if not data:
            print(f"No data to generate chart: {title}")
            return
        try:
            items = [item for item, count in data]
            counts = [count for item, count in data]
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x=counts, y=items, palette='plasma')
            plt.title(title, fontsize=16)
            plt.xlabel('Frequency', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"output/{filename}")
            plt.close()
            print(f"Chart saved to: {filename}")
        except Exception as e:
            print(f"Could not generate chart '{title}': {e}")
            
    def create_sentiment_arc_chart(data, filename):
        if not data:
            print(f"No data to generate sentiment arc chart.")
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
            plt.savefig(f"output/{filename}")
            plt.close()
            print(f"Chart saved to: {filename}")
        except Exception as e:
            print(f"Could not generate sentiment arc chart: {e}")

    create_bar_chart(word_freq.most_common(TOP_N_WORDS), 
                     f'Top {TOP_N_WORDS} Most Common Words (Lemmatized)', 
                     f"{output_base}_top_words_chart.png")

    bigram_labels = [' '.join(b) for b, c in bigram_freq.most_common(TOP_N_NGRAMS)]
    bigram_counts = [c for b, c in bigram_freq.most_common(TOP_N_NGRAMS)]
    create_bar_chart(list(zip(bigram_labels, bigram_counts)), 
                     f'Top {TOP_N_NGRAMS} Most Common Bigrams', 
                     f"{output_base}_top_bigrams_chart.png")

    if named_entities:
        entity_labels = [f"{ent[0]} ({ent[1]})" for ent, count in named_entities.most_common(TOP_N_ENTITIES)]
        entity_counts = [count for ent, count in named_entities.most_common(TOP_N_ENTITIES)]
        create_bar_chart(list(zip(entity_labels, entity_counts)), 
                         f'Top {TOP_N_ENTITIES} Named Entities', 
                         f"{output_base}_top_entities_chart.png")

    if sentiment_arc:
        create_sentiment_arc_chart(sentiment_arc, f"{output_base}_sentiment_arc.png")

def analyze_text(file_path):
    print(f"\nStarting analysis for: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    if not text.strip():
        print("The file is empty. Analysis cannot proceed.")
        return

    # --- 1. Preprocessing ---
    lemmatized_tokens = _preprocess(text)
    if not lemmatized_tokens:
        print("The text is too short or contains only stopwords. Analysis cannot proceed.")
        return
    
    sentences = sent_tokenize(text)
    original_tokens = word_tokenize(text.lower())

    # --- 2. Core Analyses ---
    word_freq = Counter(lemmatized_tokens)
    bigram_freq = Counter(ngrams(lemmatized_tokens, 2))
    trigram_freq = Counter(ngrams(lemmatized_tokens, 3))
    quadgram_freq = Counter(ngrams(lemmatized_tokens, 4))
    fivegram_freq = Counter(ngrams(lemmatized_tokens, 5))

    # --- 3. Advanced Analyses ---
    # Sentiment Analysis (Overall)
    sentiment = TextBlob(text).sentiment

    # Part-of-Speech (POS) Tagging on Lemmatized Tokens
    pos_tags = pos_tag(lemmatized_tokens)
    nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
    verbs = [word for word, tag in pos_tags if tag.startswith('VB')]
    adjectives = [word for word, tag in pos_tags if tag.startswith('JJ')]
    noun_freq = Counter(nouns)
    verb_freq = Counter(verbs)
    adjective_freq = Counter(adjectives)

    # Named Entity Recognition (NER)
    all_ents = Counter()
    if nlp:
        for chunk in split_text(text, MAX_SPACY_CHARS):
            doc_chunk = nlp(chunk)
            all_ents.update([(ent.text.strip(), ent.label_) for ent in doc_chunk.ents])
    named_entities = all_ents

    # TF-IDF Keyword Extraction
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=word_tokenize)
    try:
        tfidf_matrix = vectorizer.fit_transform(sent_tokenize(text))
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).tolist()[0]
        tfidf_word_scores = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    except ValueError:
        tfidf_word_scores = []

    # Topic Modeling (LDA)
    try:
        dictionary = corpora.Dictionary([lemmatized_tokens])
        corpus = [dictionary.doc2bow(lemmatized_tokens)]
        lda_model = LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=10, random_state=42)
        topics = lda_model.print_topics(num_words=5)
    except Exception as e:
        topics = [f"Could not perform topic modeling: {e}"]
    
    # --- 4. NEW: Automated Sense-Making Logic ---
    
    # 4a. Thematic Concept Identification
    top_nouns_set = {word for word, count in noun_freq.most_common(TOP_N_WORDS)}
    top_tfidf_set = {word for word, score in tfidf_word_scores[:TOP_N_WORDS]}
    top_entity_text_set = {ent[0].lower() for ent, count in named_entities.most_common(TOP_N_ENTITIES)}
    key_concepts = top_nouns_set.intersection(top_tfidf_set)
    entity_concepts = key_concepts.intersection(top_entity_text_set)

    # 4b. Entity-Centric Analysis (Entity Profiling)
    entity_profiles = {} 
    if nlp and named_entities:
        print("\nStarting Entity Profiling Analysis (processing text in chunks)...")
        top_entities_to_profile = [ent[0] for ent, count in named_entities.most_common(10)]
        
        aggregated_profiles = {
            entity_text: {'actions': Counter(), 'descriptors': Counter()}
            for entity_text in top_entities_to_profile
        }

        for chunk in split_text(text, MAX_SPACY_CHARS):
            doc_chunk = nlp(chunk)
            for ent in doc_chunk.ents:
                if ent.text in aggregated_profiles:
                    # Find action (verb) where entity is the subject
                    if ent.root.dep_ == 'nsubj' and ent.root.head.pos_ == 'VERB':
                        aggregated_profiles[ent.text]['actions'][ent.root.head.lemma_] += 1
                    # Find descriptor (adjective) modifying the entity
                    for child in ent.root.children:
                        if child.pos_ == 'ADJ':
                            aggregated_profiles[ent.text]['descriptors'][child.lemma_] += 1
        
        for entity, counters in aggregated_profiles.items():
            actions = counters['actions'].most_common(5)
            descriptors = counters['descriptors'].most_common(5)
            if actions or descriptors:
                entity_profiles[entity] = {'actions': actions, 'descriptors': descriptors}

    # 4c. Sentiment Arc Analysis
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
                    chunk_sentiment = TextBlob(chunk_text).sentiment.polarity
                    sentiment_arc.append(chunk_sentiment)

    # --- 5. Generate Report ---
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
    report_lines.append(f"Average Word Length: {sum(len(w) for w in original_tokens) / len(original_tokens):.2f}")

    report_lines.append("\n--- Readability Scores ---")
    report_lines.append(f"Flesch Reading Ease: {textstat.flesch_reading_ease(text):.2f} (Higher is easier)")
    report_lines.append(f"Flesch-Kincaid Grade Level: {textstat.flesch_kincaid_grade(text)}")
    report_lines.append(f"Gunning Fog Index: {textstat.gunning_fog(text):.2f}")
    report_lines.append(f"SMOG Index: {textstat.smog_index(text)}")

    report_lines.append("\n--- Sentiment Analysis (Overall) ---")
    report_lines.append(f"Polarity: {sentiment.polarity:.2f} (Ranges from -1 [negative] to +1 [positive])")
    report_lines.append(f"Subjectivity: {sentiment.subjectivity:.2f} (Ranges from 0 [objective] to 1 [subjective])")
    
    report_lines.append(f"\n--- Sentiment Arc (document split into {len(sentiment_arc)} parts) ---")
    arc_scores_str = [f"{score:.2f}" for score in sentiment_arc]
    report_lines.append(" -> ".join(arc_scores_str))

    report_lines.append(f"\n--- Discovered Topics (LDA, {NUM_TOPICS} topics) ---")
    for topic_num, topic_words in topics:
        report_lines.append(f"Topic {topic_num + 1}: {topic_words}")

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

    # --- 6. Output Results ---
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
        print("\n" + "=" * 60)
        print(f"Report saved to: {output_filepath}")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: Could not save report. {e}")

    # --- 7. Generate and Save Visualizations ---
    _generate_visualizations(basename, word_freq, bigram_freq, named_entities, sentiment_arc)
    print("\nAnalysis complete.")

    # --- 8. Claude Analysis ---
    summary = summarize_txt_file_with_claude(output_filepath)
    if summary:
        summary_filename = os.path.splitext(output_filepath)[0] + "_claude_summary.md"
        try:
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"\nClaude summary saved to: {summary_filename}")
        except Exception as e:
            print(f"\nError: Could not save Claude summary. {e}")
    else:
        print("No summary generated.")

if __name__ == "__main__":
    input_dir = "input"
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            analyze_text(file_path)