from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import TOP_N_ENTITIES, TOP_N_WORDS


def compute_tfidf_scores(sentences):
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=word_tokenize, token_pattern=None)
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).tolist()[0]
        return sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    except ValueError:
        return []


def find_key_concepts(noun_freq, tfidf_word_scores, named_entities):
    top_nouns_set = {word for word, count in noun_freq.most_common(TOP_N_WORDS)}
    top_tfidf_set = {word for word, score in tfidf_word_scores[:TOP_N_WORDS]}
    top_entity_text_set = {ent[0].lower() for ent, count in named_entities.most_common(TOP_N_ENTITIES)}
    key_concepts = top_nouns_set.intersection(top_tfidf_set)
    entity_concepts = key_concepts.intersection(top_entity_text_set)
    return key_concepts, entity_concepts
