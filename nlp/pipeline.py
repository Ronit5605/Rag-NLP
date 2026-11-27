"""
Unified NLP Pipeline orchestrating preprocessing, linguistics, sentiment/emotion,
and extractive summarization in a single pass.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from nlp.linguistics import analyze_linguistics
from nlp.preprocessing import clean_transcript, get_spacy_nlp, preprocess_text
from nlp.retrieval import extractive_summarize
from nlp.sentiment import analyze_sentiment_and_emotions


class NLPAnalyzer:
    """Orchestrates end-to-end NLP analysis on transcripts and raw text."""

    def __init__(self) -> None:
        self.nlp = get_spacy_nlp()

    def analyze(
        self,
        text: str,
        timed_items: list[dict[str, Any]] | None = None,
        num_segments: int = 6,
        summary_sentences: int = 3,
        max_token_samples: int = 100,
    ) -> dict[str, Any]:
        """Run all NLP stages on the input text and return structured analytics."""
        cleaned = clean_transcript(text)
        doc = self.nlp(cleaned)

        prep = preprocess_text(cleaned, doc=doc, max_token_samples=max_token_samples)
        ling = analyze_linguistics(cleaned, doc=doc)
        sent = analyze_sentiment_and_emotions(
            cleaned,
            timed_items=timed_items,
            num_segments=num_segments,
        )
        summary = extractive_summarize(prep["sentences"], top_k=summary_sentences)

        return {
            "cleaned_text": prep["cleaned_text"],
            "metrics": prep["metrics"],
            "preprocessing": {
                "top_tokens": prep["top_tokens"],
                "bigrams": prep["bigrams"],
                "trigrams": prep["trigrams"],
                "token_samples": prep["token_samples"],
            },
            "linguistics": ling,
            "sentiment": sent,
            "summary": summary,
        }


@lru_cache(maxsize=1)
def get_analyzer() -> NLPAnalyzer:
    """Return a cached singleton NLPAnalyzer instance."""
    return NLPAnalyzer()
