# nlp/sentiment.py
"""
VADER sentiment wrapper.

Returns (score, label)
score: compound score in [-1,1]
label: 'positive'|'neutral'|'negative'
"""
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure VADER lexicon is present
nltk.download("vader_lexicon", quiet=True)
_sid = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str):
    if not text:
        return 0.0, "neutral"
    s = _sid.polarity_scores(text)
    compound = float(s.get("compound", 0.0))
    # thresholding: common VADER thresholds
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return compound, label
