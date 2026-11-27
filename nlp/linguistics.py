"""
Linguistic analysis: Part-of-Speech (POS) distribution, Named Entity Recognition (NER),
noun chunking, and TF-IDF keyphrase extraction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp.preprocessing import clean_transcript, get_spacy_nlp

_POS_DESCRIPTIONS = {
    "NOUN": "Noun",
    "PROPN": "Proper Noun",
    "VERB": "Verb",
    "ADJ": "Adjective",
    "ADV": "Adverb",
    "PRON": "Pronoun",
    "ADP": "Adposition / Preposition",
    "DET": "Determiner",
    "CONJ": "Conjunction",
    "CCONJ": "Coordinating Conjunction",
    "SCONJ": "Subordinating Conjunction",
    "NUM": "Numeral",
    "AUX": "Auxiliary Verb",
    "PART": "Particle",
    "INTJ": "Interjection",
    "PUNCT": "Punctuation",
    "SYM": "Symbol",
    "X": "Other",
}

_ENTITY_DESCRIPTIONS = {
    "PERSON": "People, including fictional",
    "ORG": "Companies, agencies, institutions",
    "GPE": "Countries, cities, states",
    "LOC": "Non-GPE locations, regions",
    "DATE": "Dates or periods",
    "TIME": "Times smaller than a day",
    "PRODUCT": "Objects, vehicles, software, products",
    "EVENT": "Named hurricanes, battles, events",
    "WORK_OF_ART": "Titles of books, songs, papers",
    "NORP": "Nationalities, religious or political groups",
    "MONEY": "Monetary values",
    "PERCENT": "Percentage values",
    "CARDINAL": "Numerals not covered by other types",
    "ORDINAL": "First, second, etc.",
    "QUANTITY": "Measurements",
    "FAC": "Buildings, airports, highways",
    "LAW": "Named legal documents",
    "LANGUAGE": "Named languages",
}


def extract_tfidf_keywords(sentences: list[str], top_k: int = 12) -> list[dict[str, Any]]:
    """Extract top keywords and phrases across transcript sentences using TF-IDF."""
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    if not valid_sentences:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=300,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{1,}\b",
        )
        matrix = vectorizer.fit_transform(valid_sentences)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = matrix.mean(axis=0).A1
        ranked_indices = mean_scores.argsort()[::-1][:top_k]
        return [
            {
                "keyword": str(feature_names[idx]),
                "score": round(float(mean_scores[idx]), 4),
            }
            for idx in ranked_indices
            if mean_scores[idx] > 0
        ]
    except ValueError:
        return []


def analyze_linguistics(
    text: str,
    doc: spacy.tokens.Doc | None = None,
) -> dict[str, Any]:
    """Perform POS distribution, NER extraction, noun phrase chunking, and TF-IDF keyword extraction."""
    cleaned = clean_transcript(text)
    nlp = get_spacy_nlp()
    if doc is None:
        doc = nlp(cleaned)

    # POS distribution (excluding whitespace)
    pos_counts = Counter(tok.pos_ for tok in doc if not tok.is_space and tok.pos_)
    total_pos = sum(pos_counts.values()) or 1
    pos_distribution = [
        {
            "pos": pos,
            "description": _POS_DESCRIPTIONS.get(pos, pos),
            "count": count,
            "percentage": round((count / total_pos) * 100.0, 2),
        }
        for pos, count in pos_counts.most_common()
    ]

    # Named Entity Recognition
    entities_list = []
    grouped_mentions: dict[str, Counter] = defaultdict(Counter)
    for ent in doc.ents:
        ent_text = ent.text.strip()
        if not ent_text:
            continue
        entities_list.append(
            {
                "text": ent_text,
                "label": ent.label_,
                "description": _ENTITY_DESCRIPTIONS.get(ent.label_, ent.label_),
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            }
        )
        grouped_mentions[ent.label_][ent_text] += 1

    entities_by_type = []
    for label, counter in sorted(grouped_mentions.items(), key=lambda x: -sum(x[1].values())):
        entities_by_type.append(
            {
                "label": label,
                "description": _ENTITY_DESCRIPTIONS.get(label, label),
                "total_mentions": sum(counter.values()),
                "unique_entities": [
                    {"text": name, "count": cnt} for name, cnt in counter.most_common(15)
                ],
            }
        )

    # Noun chunks (when dependency parser is active)
    noun_chunk_counts: Counter = Counter()
    if doc.has_annotation("DEP"):
        for chunk in doc.noun_chunks:
            normalized = " ".join(
                tok.text.lower() for tok in chunk if not tok.is_stop and tok.is_alpha
            ).strip()
            if len(normalized) > 2:
                noun_chunk_counts[normalized] += 1

    top_noun_phrases = [
        {"phrase": phrase, "count": cnt} for phrase, cnt in noun_chunk_counts.most_common(15)
    ]

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences and cleaned:
        sentences = [cleaned]
    tfidf_keywords = extract_tfidf_keywords(sentences, top_k=12)

    return {
        "pos_distribution": pos_distribution,
        "entities": entities_list,
        "entities_by_type": entities_by_type,
        "total_entities": len(entities_list),
        "noun_phrases": top_noun_phrases,
        "tfidf_keywords": tfidf_keywords,
    }
