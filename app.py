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
TOP_N_POS = 30
NUM_TOPICS = 10
SUMMARY_SENTENCE_COUNT = 10


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
                    max_tokens=2000,
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
    
def split_text(text, max_len=500_000):
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

def _generate_visualizations(output_base, word_freq, bigram_freq, named_entities):
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
    # Sentiment Analysis
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
    for chunk in split_text(text):
        doc = nlp(chunk)
        all_ents.update([(ent.text.strip(), ent.label_) for ent in doc.ents])
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
    
    # --- 4. Generate Report ---
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"Enhanced Text Analysis Report for: {file_path}")
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

    report_lines.append("\n--- Sentiment Analysis ---")
    report_lines.append(f"Polarity: {sentiment.polarity:.2f} (Ranges from -1 [negative] to +1 [positive])")
    report_lines.append(f"Subjectivity: {sentiment.subjectivity:.2f} (Ranges from 0 [objective] to 1 [subjective])")

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

    report_lines.append(f"\n--- Top {TOP_N_NGRAMS} Most Common Trigrams (Lemmatized) ---")
    for quadgram, count in quadgram_freq.most_common(TOP_N_NGRAMS):
        report_lines.append(f"{' '.join(quadgram)}: {count}")

    report_lines.append(f"\n--- Top {TOP_N_NGRAMS} Most Common Trigrams (Lemmatized) ---")
    for fivegram, count in fivegram_freq.most_common(TOP_N_NGRAMS):
        report_lines.append(f"{' '.join(fivegram)}: {count}")

    # --- 5. Output Results ---
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

    # --- 6. Generate and Save Visualizations ---
    _generate_visualizations(basename, word_freq, bigram_freq, named_entities)
    print("\nAnalysis complete.")

    # --- 7. Claude Analysis ---
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