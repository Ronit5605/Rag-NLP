"""
Text preprocessing, normalization, tokenization, stemming, lemmatization,
n-gram extraction, and readability analysis.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Any

import nltk
import spacy
from nltk.stem import PorterStemmer

try:
    import textstat
except ImportError:  # pragma: no cover
    textstat = None


_CAPTION_NOISE_RE = re.compile(
    r"\[(?:music|applause|laughter|cheering|inaudible|silence|sound|noise|crosstalk)[^\]]*\]"
    r"|\((?:music|applause|laughter|inaudible)[^)]*\)",
    flags=re.IGNORECASE,
)
_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_WHITESPACE_RE = re.compile(r"\s+")

_STEMMER = PorterStemmer()


@lru_cache(maxsize=1)
def get_spacy_nlp() -> spacy.language.Language:
    """Load and cache the spaCy English pipeline with a lightweight fallback."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp


def clean_transcript(text: str) -> str:
    """Normalize transcript text by removing caption tags, timestamps, and extra whitespace."""
    if not text:
        return ""
    cleaned = _CAPTION_NOISE_RE.sub(" ", text)
    cleaned = _TIMESTAMP_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\u2019", "'").replace("\u2018", "'")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2014", " - ").replace("\u2013", " - ")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _has_nltk_cmudict() -> bool:
    try:
        nltk.data.find("corpora/cmudict")
        return True
    except LookupError:
        return False


def _estimate_syllables(word: str) -> int:
    """Estimate English syllable count heuristically without external corpus lookups."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    groups = re.findall(r"[aeiouy]{1,2}", w)
    return max(1, len(groups))


def _compute_readability(cleaned_text: str, word_count: int, sentence_count: int) -> dict[str, Any]:
    """Compute readability metrics using textstat (when cmudict is cached) or a pure-Python Flesch formula."""
    reading_time_min = round(word_count / 200.0, 2) if word_count > 0 else 0.0
    if textstat is not None and word_count >= 5 and _has_nltk_cmudict():
        try:
            flesch_ease = round(float(textstat.flesch_reading_ease(cleaned_text)), 2)
            fk_grade = round(float(textstat.flesch_kincaid_grade(cleaned_text)), 2)
            return {
                "flesch_reading_ease": flesch_ease,
                "flesch_kincaid_grade": fk_grade,
                "estimated_reading_minutes": reading_time_min,
            }
        except Exception:
            pass

    words = re.findall(r"[a-zA-Z]+", cleaned_text)
    total_words = max(len(words), 1)
    total_sentences = max(sentence_count, 1)
    total_syllables = sum(_estimate_syllables(w) for w in words)

    avg_sentence_len = total_words / total_sentences
    avg_syllables_per_word = total_syllables / total_words

    flesch_ease = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables_per_word
    fk_grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables_per_word - 15.59

    return {
        "flesch_reading_ease": round(max(0.0, min(100.0, flesch_ease)), 2),
        "flesch_kincaid_grade": round(max(0.0, fk_grade), 2),
        "estimated_reading_minutes": reading_time_min,
    }


def extract_ngrams(tokens: list[str], n: int, top_k: int = 12) -> list[dict[str, Any]]:
    """Return top_k n-grams and their frequencies from a list of normalized tokens."""
    if len(tokens) < n or n < 1:
        return []
    grams = (" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    counts = Counter(grams)
    return [{"ngram": gram, "count": count} for gram, count in counts.most_common(top_k)]


def preprocess_text(
    text: str,
    doc: spacy.tokens.Doc | None = None,
    max_token_samples: int = 100,
) -> dict[str, Any]:
    """Run full preprocessing: cleaning, sentence splitting, tokenization, stemming, lemmatization, and n-grams."""
    cleaned = clean_transcript(text)
    nlp = get_spacy_nlp()
    if doc is None:
        doc = nlp(cleaned)

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences and cleaned:
        sentences = [cleaned]

    alpha_tokens = [tok for tok in doc if tok.is_alpha]
    content_lemmas = [
        tok.lemma_.lower()
        for tok in alpha_tokens
        if not tok.is_stop and len(tok.lemma_) > 1 and tok.lemma_ != "-PRON-"
    ]

    word_count = len(alpha_tokens)
    unique_lemmas = set(content_lemmas)
    lexical_diversity = round(len(unique_lemmas) / word_count, 4) if word_count > 0 else 0.0

    token_samples = []
    for tok in doc:
        if tok.is_space:
            continue
        token_samples.append(
            {
                "text": tok.text,
                "lemma": tok.lemma_,
                "stem": _STEMMER.stem(tok.text.lower()) if tok.is_alpha else tok.text,
                "pos": tok.pos_ or "X",
                "tag": tok.tag_ or "X",
                "is_stop": bool(tok.is_stop),
                "is_alpha": bool(tok.is_alpha),
            }
        )
        if len(token_samples) >= max_token_samples:
            break

    lemma_counts = Counter(content_lemmas)
    top_tokens = [{"token": tok, "count": cnt} for tok, cnt in lemma_counts.most_common(20)]

    readability = _compute_readability(cleaned, word_count, len(sentences))

    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "metrics": {
            "char_count": len(cleaned),
            "word_count": word_count,
            "sentence_count": len(sentences),
            "unique_content_lemmas": len(unique_lemmas),
            "lexical_diversity_ttr": lexical_diversity,
            **readability,
        },
        "top_tokens": top_tokens,
        "bigrams": extract_ngrams(content_lemmas, n=2, top_k=12),
        "trigrams": extract_ngrams(content_lemmas, n=3, top_k=10),
        "token_samples": token_samples,
    }
