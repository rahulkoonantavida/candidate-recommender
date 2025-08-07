import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# check for—and if missing, download—the punkt tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# check for—and if missing, download—the stopwords corpus
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

def clean_text(raw: str) -> str:
    # Remove URLs
    text = re.sub(r'https?://\S+', '', raw)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove phone numbers (simple U.S. pattern)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
    # Remove “Page X of Y” footers
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    # Squash multiple blank lines
    text = re.sub(r'\n{2,}', '\n\n', text)
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text) 
    # lowercase
    text = text.lower()
    # Tokenize the text
    words = word_tokenize(text)
    # Get English stop words
    stop_words = set(stopwords.words('english'))
    # Filter out stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Join the filtered words back into a sentence
    clean_text = " ".join(filtered_words)
    return clean_text