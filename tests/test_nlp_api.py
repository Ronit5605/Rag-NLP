"""
Unit and FastAPI integration tests for the YouTube NLP & RAG Studio.

Run:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main
from nlp.linguistics import analyze_linguistics
from nlp.pipeline import get_analyzer
from nlp.preprocessing import clean_transcript, preprocess_text
from nlp.retrieval import extractive_summarize, retrieve_top_chunks_tfidf
from nlp.sentiment import analyze_sentiment_and_emotions, score_emotions, score_sentiment


SAMPLE_TEXT = (
    "[Music] 00:12 Welcome to Google DeepMind Summit in London! "
    "Today Alice and Dr. Smith demonstrate how spaCy and FastAPI power modern NLP pipelines. "
    "The initial prototype was slow and frustrating, but our latest release is fast, accurate, and awesome! "
    "Why do developers love transparent entity extraction? Because it builds trust."
)


class TestNLPModules(unittest.TestCase):
    def test_clean_transcript_removes_noise_and_timestamps(self):
        raw = "[Music] Hello   01:23 world (laughter)!"
        cleaned = clean_transcript(raw)
        self.assertEqual(cleaned, "Hello world !")

    def test_preprocessing_outputs_tokens_lemmas_stems_and_ngrams(self):
        result = preprocess_text(SAMPLE_TEXT)
        self.assertGreater(result["metrics"]["word_count"], 25)
        self.assertGreater(result["metrics"]["sentence_count"], 2)
        self.assertTrue(0.0 < result["metrics"]["lexical_diversity_ttr"] <= 1.0)
        self.assertTrue(len(result["token_samples"]) > 10)
        sample_first = result["token_samples"][0]
        self.assertIn("lemma", sample_first)
        self.assertIn("stem", sample_first)
        self.assertIn("pos", sample_first)

    def test_linguistics_extracts_pos_and_entities(self):
        result = analyze_linguistics(SAMPLE_TEXT)
        self.assertGreater(len(result["pos_distribution"]), 0)
        self.assertGreater(result["total_entities"], 0)
        labels = {ent["label"] for ent in result["entities"]}
        self.assertTrue(bool(labels & {"ORG", "GPE", "PERSON"}))
        self.assertGreater(len(result["tfidf_keywords"]), 0)

    def test_sentiment_handles_negation_and_segments(self):
        pos = score_sentiment("This workflow is super clear and awesome!")
        neg = score_sentiment("This workflow is not good and completely broken.")
        self.assertEqual(pos["label"], "Positive")
        self.assertGreater(pos["polarity"], 0.2)
        self.assertEqual(neg["label"], "Negative")
        self.assertLess(neg["polarity"], -0.1)

        emo = score_emotions("I am worried about the security risks and limitations.")
        self.assertEqual(emo["dominant_emotion"], "concern")

        seg_result = analyze_sentiment_and_emotions(SAMPLE_TEXT, num_segments=3)
        self.assertEqual(len(seg_result["segments"]), 3)
        self.assertIn("polarity", seg_result["segments"][0])
        self.assertIn("dominant_emotion", seg_result["segments"][0])

    def test_extractive_summary_and_tfidf_retrieval(self):
        prep = preprocess_text(SAMPLE_TEXT)
        summary = extractive_summarize(prep["sentences"], top_k=2)
        self.assertEqual(len(summary), 2)
        hits = retrieve_top_chunks_tfidf(prep["sentences"], "London summit", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("London", hits[0]["text"])

    def test_unified_analyzer(self):
        analyzer = get_analyzer()
        full = analyzer.analyze(SAMPLE_TEXT, num_segments=3, summary_sentences=2)
        for key in ("cleaned_text", "metrics", "preprocessing", "linguistics", "sentiment", "summary"):
            self.assertIn(key, full)


class TestFastAPIEndpoints(unittest.TestCase):
    @staticmethod
    def _request(method: str, path: str, **kwargs) -> httpx.Response:
        async def _call():
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(_call())

    def test_dashboard_and_health_and_samples(self):
        root_res = self._request("GET", "/")
        self.assertEqual(root_res.status_code, 200)
        self.assertIn("YouTube NLP", root_res.text)

        health_res = self._request("GET", "/health")
        self.assertEqual(health_res.status_code, 200)
        self.assertEqual(health_res.json()["status"], "ok")

        samples_res = self._request("GET", "/nlp/samples")
        self.assertEqual(samples_res.status_code, 200)
        self.assertGreaterEqual(len(samples_res.json()["samples"]), 2)

    def test_nlp_analyze_preprocess_entities_sentiment_endpoints(self):
        payload = {"text": SAMPLE_TEXT, "num_segments": 3, "summary_sentences": 2}

        analyze_res = self._request("POST", "/nlp/analyze", json=payload)
        self.assertEqual(analyze_res.status_code, 200)
        data = analyze_res.json()
        self.assertEqual(data["source"], "text")
        self.assertEqual(len(data["sentiment"]["segments"]), 3)
        self.assertEqual(len(data["summary"]), 2)

        prep_res = self._request("POST", "/nlp/preprocess", json={"text": SAMPLE_TEXT})
        self.assertEqual(prep_res.status_code, 200)
        self.assertIn("token_samples", prep_res.json())

        ent_res = self._request("POST", "/nlp/entities", json={"text": SAMPLE_TEXT})
        self.assertEqual(ent_res.status_code, 200)
        self.assertIn("entities_by_type", ent_res.json())

        sent_res = self._request("POST", "/nlp/sentiment", json={"text": SAMPLE_TEXT, "num_segments": 2})
        self.assertEqual(sent_res.status_code, 200)
        self.assertEqual(len(sent_res.json()["segments"]), 2)

    def test_validation_rejects_empty_input(self):
        res = self._request("POST", "/nlp/analyze", json={})
        self.assertEqual(res.status_code, 400)

    def test_ask_endpoint_with_mocked_rag_stack(self):
        fake_doc = MagicMock()
        fake_doc.page_content = "FastAPI powers the backend and spaCy handles NER."
        fake_retriever = MagicMock()
        fake_retriever.invoke.return_value = [fake_doc]
        fake_vectorstore = MagicMock()
        fake_vectorstore.as_retriever.return_value = fake_retriever

        fake_llm_instance = MagicMock()
        fake_llm_instance.invoke.return_value = MagicMock(content="FastAPI powers the backend.")

        main.current_video_id = None
        main.current_retriever = None

        with (
            patch.object(main, "load_video_transcript", return_value="FastAPI powers the backend and spaCy handles NER."),
            patch.object(main, "OllamaEmbeddings", return_value=MagicMock()),
            patch.object(main, "Chroma") as mock_chroma,
            patch.object(main, "ChatOllama", return_value=fake_llm_instance),
        ):
            mock_chroma.from_texts.return_value = fake_vectorstore
            res = self._request(
                "POST",
                "/ask",
                json={"video_id": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "question": "What powers the backend?"},
            )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"answer": "FastAPI powers the backend."})


if __name__ == "__main__":
    unittest.main()
