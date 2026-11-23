from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


def overall_sentiment(text: str):
    return TextBlob(text).sentiment


def compute_sentiment_arc(sentences, num_chunks: int = 20) -> list[float]:
    sentiment_arc = []
    if not sentences:
        return sentiment_arc

    sia = SentimentIntensityAnalyzer()
    chunk_size = max(1, len(sentences) // num_chunks)
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else len(sentences)
        if start < len(sentences):
            chunk_text = " ".join(sentences[start:end])
            if chunk_text:
                compound = sia.polarity_scores(chunk_text)['compound']
                sentiment_arc.append(compound)
    return sentiment_arc
