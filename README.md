# YouTube NLP & RAG Studio

An end-to-end **Natural Language Processing (NLP)** and **Retrieval-Augmented Generation (RAG)** project built on **FastAPI**, **spaCy**, **NLTK**, **scikit-learn**, and **Ollama/ChromaDB**, complete with an interactive Web Dashboard served directly from FastAPI.

---

## Key Features

1. **Text Preprocessing & Normalization (`nlp/preprocessing.py`)**
   - Caption noise removal (`[Music]`, `[Applause]`, `(laughter)`) and timestamp stripping
   - Sentence segmentation, tokenization, stopword filtering, **lemmatization** (spaCy), and **Porter stemming** (NLTK)
   - Unigram, bigram, and trigram frequency analysis
   - Vocabulary richness (**Type-Token Ratio**) and readability metrics (**Flesch Reading Ease**, **Flesch-Kincaid Grade**, reading time)

2. **Linguistic Analysis & Named Entity Recognition (`nlp/linguistics.py`)**
   - Part-of-Speech (**POS**) tagging distribution
   - **Named Entity Recognition (NER)** with entity grouping (`PERSON`, `ORG`, `GPE`, `DATE`, etc.) and character offsets for inline highlighting
   - Noun-phrase chunking and **TF-IDF keyphrase extraction**

3. **Segment-Level Sentiment & Emotion Analysis (`nlp/sentiment.py`)**
   - Sentiment polarity (`-1.0` to `+1.0`) and `Positive` / `Neutral` / `Negative` classification with negation & intensifier handling (uses NLTK VADER if `vader_lexicon` is installed, with an offline VADER-style engine fallback)
   - Multi-category **Emotion Scoring** (`joy`, `trust`, `curiosity`, `concern`, `frustration`, `sadness`)
   - Timeline segmentation across YouTube timestamps or sentence chunks

4. **Extractive Summarization & Classical Retrieval (`nlp/retrieval.py`)**
   - TF-IDF centroid salience extractive summarization preserving narrative order
   - Classical TF-IDF cosine-similarity chunk retrieval helper

5. **YouTube RAG ChatBot (`POST /ask`)**
   - Preserves the original YouTube transcript RAG endpoint using `youtube-transcript-api`, `RecursiveCharacterTextSplitter`, `Chroma` (MMR retrieval with `nomic-embed-text`), and `ChatOllama` (`llama3.2`)

6. **Interactive Web Dashboard (`static/`)**
   - Served directly at `GET /` with KPI cards, inline NER highlighting, SVG sentiment polarity timeline chart, emotion distribution bars, token inspector table, built-in sample transcripts, and RAG chat tab

---

## Project Structure

```text
Task3/
├── main.py                  # FastAPI server, RAG /ask endpoint, NLP endpoints & static UI serving
├── requirement.txt          # Python dependencies
├── README.md                # Project documentation
├── nlp/
│   ├── __init__.py
│   ├── pipeline.py          # Unified single-pass NLPAnalyzer orchestrator
│   ├── preprocessing.py     # Cleaning, tokenization, stemming, lemmatization, n-grams, readability
│   ├── linguistics.py       # POS distribution, NER, noun chunks, TF-IDF keywords
│   ├── sentiment.py         # Segment-wise sentiment polarity & emotion analysis
│   └── retrieval.py         # Extractive summarization & TF-IDF chunk retrieval
├── static/
│   ├── index.html           # Interactive NLP & RAG Studio UI
│   ├── styles.css           # Dashboard styling
│   └── app.js               # Client-side rendering & inline SVG charts
├── eval/
│   └── evaluate_nlp.py      # Quantitative evaluation (Accuracy, Macro-F1, ROUGE-1/2/L, Recall@K, MRR)
└── tests/
    └── test_nlp_api.py      # Unit and FastAPI TestClient integration tests
```

---

## Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirement.txt
python -m spacy download en_core_web_sm
```

*(Optional for the `/ask` RAG ChatBot endpoint: ensure [Ollama](https://ollama.com/) is running with `nomic-embed-text` and `llama3.2` pulled.)*

### 2. Start the FastAPI Server & Web UI
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open **http://127.0.0.1:8000** in your browser for the interactive NLP Studio, or **http://127.0.0.1:8000/docs** for Swagger API documentation.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Interactive NLP & RAG Web Dashboard |
| `/health` | `GET` | Service status, loaded spaCy model, and RAG dependency availability |
| `/nlp/samples` | `GET` | Built-in timestamped sample transcripts |
| `/nlp/analyze` | `POST` | Full NLP pipeline (`text` or `video_id`) |
| `/nlp/preprocess` | `POST` | Cleaning, tokenization, lemmatization, stemming, n-grams, readability |
| `/nlp/entities` | `POST` | POS distribution, Named Entity Recognition, noun phrases, TF-IDF keywords |
| `/nlp/sentiment` | `POST` | Overall and segment-wise sentiment polarity & emotion timeline |
| `/ask` | `POST` | YouTube transcript RAG ChatBot (`{ "video_id": "...", "question": "..." }`) |

### Example Request (`POST /nlp/analyze`)
```bash
curl -X POST http://127.0.0.1:8000/nlp/analyze \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Welcome to the AI Summit in London! Today spaCy and FastAPI power transparent NLP pipelines.\", \"num_segments\": 2}"
```

---

## Testing & Evaluation

Run the unit and API integration tests:
```bash
python -m unittest discover -s tests -v
```

Run the quantitative NLP benchmark (`Accuracy`, `Macro-F1`, `ROUGE-1/2/L`, `Recall@K`, `MRR`):
```bash
python eval/evaluate_nlp.py
```
