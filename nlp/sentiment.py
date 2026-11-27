"""
Segment-level Sentiment Polarity and Emotion Analysis.

Uses NLTK VADER when `vader_lexicon` is available locally, and falls back to a
deterministic, offline negation- and intensifier-aware sentiment & emotion engine
so analysis works reliably without external downloads.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from nlp.preprocessing import clean_transcript, get_spacy_nlp

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    SentimentIntensityAnalyzer = None


_POSITIVE_WORDS: dict[str, float] = {
    "amazing": 2.4, "awesome": 2.3, "excellent": 2.5, "fantastic": 2.5, "great": 2.0,
    "good": 1.6, "wonderful": 2.4, "brilliant": 2.3, "outstanding": 2.5, "superb": 2.4,
    "love": 2.2, "loved": 2.2, "loving": 2.0, "like": 1.1, "enjoy": 1.8, "enjoyed": 1.8,
    "helpful": 1.7, "useful": 1.5, "effective": 1.6, "efficient": 1.6, "powerful": 1.8,
    "clean": 1.3, "clear": 1.4, "easy": 1.4, "intuitive": 1.7, "fast": 1.4,
    "impressive": 2.0, "incredible": 2.2, "exciting": 2.0, "excited": 1.9, "happy": 2.0,
    "glad": 1.6, "grateful": 1.9, "thank": 1.4, "thanks": 1.5, "best": 2.2,
    "better": 1.4, "improve": 1.4, "improved": 1.6, "improvement": 1.5, "success": 2.0,
    "successful": 2.0, "win": 1.9, "winning": 1.8, "benefit": 1.5, "benefits": 1.5,
    "advantage": 1.5, "accurate": 1.7, "reliable": 1.7, "robust": 1.6, "scalable": 1.5,
    "innovative": 1.8, "smart": 1.5, "elegant": 1.8, "smooth": 1.4, "solid": 1.4,
    "positive": 1.6, "confident": 1.7, "recommend": 1.7, "recommended": 1.8, "favorite": 1.9,
    "perfect": 2.4, "ideal": 1.8, "delightful": 2.1, "valuable": 1.7, "insightful": 1.8,
    "bright": 1.4, "promising": 1.6, "strong": 1.4, "clarity": 1.5, "achieve": 1.5,
    "achieved": 1.6, "solved": 1.6, "solution": 1.3, "progress": 1.4, "growth": 1.4,
    "loveable": 2.0, "cool": 1.3, "neat": 1.2, "wow": 1.9, "yay": 2.0,
}

_NEGATIVE_WORDS: dict[str, float] = {
    "bad": -1.8, "terrible": -2.5, "horrible": -2.5, "awful": -2.4, "worst": -2.6,
    "poor": -1.7, "hate": -2.4, "hated": -2.3, "dislike": -1.7, "annoying": -1.8,
    "frustrating": -2.0, "frustrated": -2.0, "broken": -1.9, "fail": -1.9, "failed": -2.0,
    "failure": -2.1, "bug": -1.4, "bugs": -1.5, "error": -1.4, "errors": -1.5,
    "crash": -2.0, "crashed": -2.1, "slow": -1.4, "sluggish": -1.6, "confusing": -1.7,
    "confused": -1.6, "difficult": -1.3, "hard": -0.9, "complicated": -1.3, "complex": -0.8,
    "problem": -1.4, "problems": -1.5, "issue": -1.2, "issues": -1.3, "risk": -1.3,
    "risky": -1.5, "danger": -1.9, "dangerous": -2.0, "wrong": -1.6, "mistake": -1.6,
    "disappointing": -2.0, "disappointed": -2.0, "useless": -2.2, "worthless": -2.4,
    "weak": -1.5, "flaw": -1.6, "flawed": -1.7, "limitation": -1.2, "limitations": -1.3,
    "struggle": -1.5, "struggling": -1.6, "painful": -1.9, "boring": -1.6, "ugly": -1.8,
    "sad": -1.9, "unhappy": -1.9, "worried": -1.6, "worry": -1.5, "concern": -1.2,
    "concerned": -1.4, "fear": -1.8, "afraid": -1.8, "angry": -2.1, "upset": -1.8,
    "expensive": -1.2, "waste": -1.8, "wasted": -1.9, "missing": -1.2, "lack": -1.3,
    "lacking": -1.4, "unavailable": -1.4, "impossible": -1.7, "unfortunately": -1.3,
    "negative": -1.6, "toxic": -2.2, "messy": -1.5, "chaotic": -1.7, "stuck": -1.4,
    "regret": -1.8, "loss": -1.6, "lost": -1.4, "decline": -1.4, "drop": -1.0,
}

_NEGATIONS = {
    "not", "no", "never", "neither", "nor", "none", "nobody", "nothing", "nowhere",
    "hardly", "scarcely", "barely", "dont", "don't", "doesnt", "doesn't", "didnt",
    "didn't", "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't", "werent",
    "weren't", "wont", "won't", "cant", "can't", "cannot", "couldnt", "couldn't",
    "shouldnt", "shouldn't", "wouldnt", "wouldn't", "without",
}

_INTENSIFIERS: dict[str, float] = {
    "very": 1.35, "really": 1.3, "extremely": 1.5, "incredibly": 1.5, "super": 1.4,
    "absolutely": 1.45, "completely": 1.4, "totally": 1.35, "highly": 1.35,
    "deeply": 1.35, "especially": 1.25, "particularly": 1.25, "remarkably": 1.35,
    "slightly": 0.65, "somewhat": 0.7, "barely": 0.5, "a bit": 0.7, "less": 0.6,
}

_EMOTION_LEXICON: dict[str, set[str]] = {
    "joy": {
        "happy", "joy", "excited", "exciting", "amazing", "awesome", "love", "loved",
        "wonderful", "fantastic", "great", "enjoy", "enjoyed", "delightful", "fun",
        "celebrate", "thrilled", "glad", "smile", "cheerful", "brilliant", "yay", "wow",
    },
    "trust": {
        "reliable", "accurate", "trust", "trusted", "confident", "solid", "proven",
        "safe", "secure", "stable", "consistent", "verified", "clear", "helpful",
        "effective", "robust", "recommend", "recommended", "guarantee", "dependable",
    },
    "curiosity": {
        "why", "how", "wonder", "wondering", "curious", "explore", "exploring",
        "discover", "question", "interesting", "fascinating", "intriguing", "imagine",
        "investigate", "understand", "learn", "learning", "experiment", "notice",
    },
    "concern": {
        "worry", "worried", "concern", "concerned", "risk", "risky", "danger",
        "warning", "careful", "caution", "afraid", "fear", "uncertain", "threat",
        "vulnerability", "limitation", "challenge", "problem", "issue", "struggle",
    },
    "frustration": {
        "frustrating", "frustrated", "annoying", "broken", "fail", "failed", "failure",
        "crash", "crashed", "slow", "bug", "bugs", "error", "terrible", "awful",
        "useless", "hate", "angry", "stuck", "waste", "wasted", "horrible", "worst",
    },
    "sadness": {
        "sad", "unhappy", "disappointed", "disappointing", "unfortunately", "regret",
        "loss", "lost", "miss", "missing", "lonely", "sorry", "tragic", "painful",
    },
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _load_vader_if_available() -> Any | None:
    if SentimentIntensityAnalyzer is None:
        return None
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        return None


_VADER_ANALYZER = _load_vader_if_available()


def _rule_based_sentiment(text: str) -> dict[str, float]:
    """Compute VADER-style polarity scores offline using lexicon, negation, and intensifiers."""
    tokens = [tok.lower() for tok in _WORD_RE.findall(text)]
    if not tokens:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

    sentiments: list[float] = []
    pos_sum = 0.0
    neg_sum = 0.0
    neu_count = 0

    for idx, word in enumerate(tokens):
        valence = _POSITIVE_WORDS.get(word) or _NEGATIVE_WORDS.get(word)
        if valence is None:
            neu_count += 1
            continue

        # Apply preceding intensifier (1-2 tokens back)
        for back in (1, 2):
            if idx - back >= 0 and tokens[idx - back] in _INTENSIFIERS:
                valence *= _INTENSIFIERS[tokens[idx - back]]
                break

        # Apply negation window (up to 3 tokens back)
        window_start = max(0, idx - 3)
        if any(tokens[j] in _NEGATIONS for j in range(window_start, idx)):
            valence *= -0.74

        # Boost slightly after contrastive "but" / "however"
        if any(tokens[j] in {"but", "however", "yet"} for j in range(max(0, idx - 6), idx)):
            valence *= 1.15

        sentiments.append(valence)
        if valence > 0:
            pos_sum += valence
        elif valence < 0:
            neg_sum += abs(valence)

    if not sentiments:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

    raw_sum = sum(sentiments)
    exclamations = min(text.count("!"), 4) * 0.25
    if raw_sum > 0:
        raw_sum += exclamations
    elif raw_sum < 0:
        raw_sum -= exclamations

    compound = raw_sum / math.sqrt(raw_sum * raw_sum + 15.0)
    compound = round(max(-1.0, min(1.0, compound)), 4)

    total_mass = pos_sum + neg_sum + float(neu_count)
    pos_ratio = round(pos_sum / total_mass, 3) if total_mass else 0.0
    neg_ratio = round(neg_sum / total_mass, 3) if total_mass else 0.0
    neu_ratio = round(max(0.0, 1.0 - pos_ratio - neg_ratio), 3)

    return {"compound": compound, "pos": pos_ratio, "neu": neu_ratio, "neg": neg_ratio}


def score_sentiment(text: str) -> dict[str, Any]:
    """Return sentiment label, polarity score [-1, 1], and pos/neu/neg breakdown."""
    if not text.strip():
        return {
            "label": "Neutral",
            "polarity": 0.0,
            "scores": {"pos": 0.0, "neu": 1.0, "neg": 0.0},
            "engine": "offline-lexicon",
        }

    if _VADER_ANALYZER is not None:
        scores = _VADER_ANALYZER.polarity_scores(text)
        compound = round(float(scores["compound"]), 4)
        engine = "nltk-vader"
        breakdown = {
            "pos": round(float(scores["pos"]), 3),
            "neu": round(float(scores["neu"]), 3),
            "neg": round(float(scores["neg"]), 3),
        }
    else:
        scores = _rule_based_sentiment(text)
        compound = scores["compound"]
        engine = "offline-lexicon"
        breakdown = {"pos": scores["pos"], "neu": scores["neu"], "neg": scores["neg"]}

    if compound >= 0.08:
        label = "Positive"
    elif compound <= -0.08:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label": label,
        "polarity": compound,
        "scores": breakdown,
        "engine": engine,
    }


def score_emotions(text: str, polarity: float | None = None) -> dict[str, Any]:
    """Score 6 emotion categories for a text segment and determine dominant emotion."""
    tokens = [tok.lower() for tok in _WORD_RE.findall(text)]
    counts: Counter[str] = Counter()

    for idx, word in enumerate(tokens):
        negated = any(tokens[j] in _NEGATIONS for j in range(max(0, idx - 2), idx))
        for emotion, vocab in _EMOTION_LEXICON.items():
            if word in vocab:
                if negated:
                    # Negating a positive emotion mildly shifts toward concern/sadness
                    if emotion in {"joy", "trust"}:
                        counts["concern"] += 0.5
                else:
                    counts[emotion] += 1.0

    if "?" in text:
        counts["curiosity"] += min(text.count("?") * 0.75, 2.0)

    if polarity is None:
        polarity = score_sentiment(text)["polarity"]

    # Soft prior from polarity if no explicit emotion words matched
    if sum(counts.values()) == 0:
        if polarity >= 0.2:
            counts["joy"] += 0.6
            counts["trust"] += 0.4
        elif polarity <= -0.2:
            counts["concern"] += 0.6
            counts["frustration"] += 0.4
        else:
            counts["trust"] += 0.5
            counts["curiosity"] += 0.5

    total = sum(counts.values()) or 1.0
    distribution = {
        emotion: round(counts.get(emotion, 0.0) / total, 4)
        for emotion in _EMOTION_LEXICON
    }
    dominant_emotion = max(distribution.items(), key=lambda item: item[1])[0]

    return {
        "dominant_emotion": dominant_emotion,
        "emotion_scores": distribution,
    }


def _format_timestamp(seconds: float) -> str:
    total_sec = max(0, int(seconds))
    mins, secs = divmod(total_sec, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def build_segments_from_timed_items(
    timed_items: list[dict[str, Any]],
    num_segments: int = 8,
) -> list[dict[str, Any]]:
    """Group timestamped YouTube transcript snippets into evenly sized segments."""
    if not timed_items:
        return []

    target = max(1, min(num_segments, len(timed_items)))
    chunk_size = math.ceil(len(timed_items) / target)
    segments = []
    for idx in range(0, len(timed_items), chunk_size):
        group = timed_items[idx : idx + chunk_size]
        text = clean_transcript(" ".join(item.get("text", "") for item in group))
        if not text:
            continue
        start_sec = float(group[0].get("start", 0.0))
        last = group[-1]
        end_sec = float(last.get("start", start_sec)) + float(last.get("duration", 0.0))
        segments.append(
            {
                "segment_index": len(segments) + 1,
                "label": f"{_format_timestamp(start_sec)} - {_format_timestamp(end_sec)}",
                "start_seconds": round(start_sec, 2),
                "end_seconds": round(end_sec, 2),
                "text": text,
            }
        )
    return segments


def build_segments_from_text(text: str, num_segments: int = 6) -> list[dict[str, Any]]:
    """Split raw text into ordered sentence-based segments."""
    cleaned = clean_transcript(text)
    if not cleaned:
        return []

    nlp = get_spacy_nlp()
    doc = nlp(cleaned)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        sentences = [cleaned]

    target = max(1, min(num_segments, len(sentences)))
    chunk_size = math.ceil(len(sentences) / target)
    segments = []
    for idx in range(0, len(sentences), chunk_size):
        seg_text = " ".join(sentences[idx : idx + chunk_size]).strip()
        if not seg_text:
            continue
        seg_num = len(segments) + 1
        segments.append(
            {
                "segment_index": seg_num,
                "label": f"Segment {seg_num}",
                "start_seconds": None,
                "end_seconds": None,
                "text": seg_text,
            }
        )
    return segments


def analyze_sentiment_and_emotions(
    text: str,
    timed_items: list[dict[str, Any]] | None = None,
    num_segments: int = 6,
) -> dict[str, Any]:
    """Run overall and segment-by-segment sentiment polarity and emotion analysis."""
    cleaned = clean_transcript(text)
    overall_sentiment = score_sentiment(cleaned)
    overall_emotions = score_emotions(cleaned, polarity=overall_sentiment["polarity"])

    if timed_items:
        raw_segments = build_segments_from_timed_items(timed_items, num_segments=num_segments)
    else:
        raw_segments = build_segments_from_text(cleaned, num_segments=num_segments)

    analyzed_segments = []
    aggregate_emotions: Counter[str] = Counter()

    for seg in raw_segments:
        seg_sent = score_sentiment(seg["text"])
        seg_emo = score_emotions(seg["text"], polarity=seg_sent["polarity"])
        for emo, val in seg_emo["emotion_scores"].items():
            aggregate_emotions[emo] += val

        preview = seg["text"][:180] + ("..." if len(seg["text"]) > 180 else "")
        analyzed_segments.append(
            {
                "segment_index": seg["segment_index"],
                "label": seg["label"],
                "start_seconds": seg["start_seconds"],
                "end_seconds": seg["end_seconds"],
                "text_preview": preview,
                "word_count": len(_WORD_RE.findall(seg["text"])),
                "sentiment_label": seg_sent["label"],
                "polarity": seg_sent["polarity"],
                "sentiment_scores": seg_sent["scores"],
                "dominant_emotion": seg_emo["dominant_emotion"],
                "emotion_scores": seg_emo["emotion_scores"],
            }
        )

    seg_count = len(analyzed_segments) or 1
    avg_emotions = {
        emo: round(aggregate_emotions[emo] / seg_count, 4) for emo in _EMOTION_LEXICON
    }

    return {
        "overall": {
            "label": overall_sentiment["label"],
            "polarity": overall_sentiment["polarity"],
            "scores": overall_sentiment["scores"],
            "dominant_emotion": overall_emotions["dominant_emotion"],
            "emotion_scores": overall_emotions["emotion_scores"],
            "average_segment_emotions": avg_emotions,
            "engine": overall_sentiment["engine"],
        },
        "segments": analyzed_segments,
    }
