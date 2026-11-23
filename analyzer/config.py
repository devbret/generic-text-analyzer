import os

from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

INPUT_DIR = "input"
OUTPUT_DIR = "output"

TOP_N_WORDS = 50
TOP_N_NGRAMS = 50
TOP_N_ENTITIES = 50
TOP_N_POS = 50
NUM_TOPICS = 50
SUMMARY_SENTENCE_COUNT = 43
MAX_SPACY_CHARS = 250_000

CUSTOM_STOPWORDS = {
    "uh", "um", "like", "yeah", "im", "dont", "go", "know",
    "going", "thats", "think", "let", "lets",
}
