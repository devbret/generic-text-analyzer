import logging
from collections import Counter

from .config import MAX_SPACY_CHARS
from .nlp_resources import nlp
from .preprocessing import split_text

logger = logging.getLogger(__name__)


def extract_named_entities(text: str) -> Counter:
    all_ents = Counter()
    if nlp:
        for chunk in split_text(text, MAX_SPACY_CHARS):
            doc_chunk = nlp(chunk)
            all_ents.update([(ent.text.strip(), ent.label_) for ent in doc_chunk.ents])
    return all_ents


def build_entity_profiles(text: str, named_entities: Counter, top_n: int = 10) -> dict:
    entity_profiles = {}
    if not (nlp and named_entities):
        return entity_profiles

    logger.info("Starting entity profiling analysis (processing text in chunks)...")
    top_entities_to_profile = [ent[0] for ent, count in named_entities.most_common(top_n)]

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

    return entity_profiles
