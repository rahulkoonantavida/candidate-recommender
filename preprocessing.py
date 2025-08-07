import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
    # Tokenize the text
    words = word_tokenize(text)
    # Get English stop words
    stop_words = set(stopwords.words('english'))
    # Filter out stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Join the filtered words back into a sentence
    clean_text = " ".join(filtered_words)
    return clean_text