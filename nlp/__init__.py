"""
NLP Analysis Package for YouTube Transcript & Text Analytics.

Provides modular pipelines for:
- Text cleaning, tokenization, stemming, lemmatization, n-grams, and readability
- Linguistic analysis (POS tagging, Named Entity Recognition, TF-IDF keyphrases)
- Segment-level sentiment polarity and emotion analysis
- Extractive summarization and classical TF-IDF retrieval
"""

from nlp.pipeline import NLPAnalyzer, get_analyzer

__all__ = ["NLPAnalyzer", "get_analyzer"]
