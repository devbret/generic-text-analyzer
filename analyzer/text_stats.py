import textstat


def average_word_length(tokens) -> float:
    alpha_tokens = [t for t in tokens if t.isalpha()]
    return (sum(map(len, alpha_tokens)) / len(alpha_tokens)) if alpha_tokens else 0


def readability_scores(text: str) -> dict:
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
    }
