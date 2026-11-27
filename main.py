from pathlib import Path
import logging
import re
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter

from nlp import get_analyzer
from nlp.linguistics import analyze_linguistics
from nlp.preprocessing import get_spacy_nlp, preprocess_text
from nlp.sentiment import analyze_sentiment_and_emotions

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

try:
    from langchain_chroma import Chroma
except ImportError:
    Chroma = None

try:
    from langchain_ollama import OllamaEmbeddings, ChatOllama
except ImportError:
    OllamaEmbeddings = None
    ChatOllama = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("youtube-rag-backend")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="YouTube NLP & RAG Studio",
    description="YouTube Transcript RAG ChatBot expanded with an NLP Analytics Pipeline (Preprocessing, POS, NER, Sentiment & Emotion Timeline, Extractive Summarization).",
    version="2.0.0",
)

# Simple CORS configuration for local development.
# WARNING: allow_origins=["*"] is configured to allow the unpacked Chrome extension
# (which runs with dynamic extension origins like chrome-extension://<id>) to connect locally.
# This should not be used in a production setup.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_YOUTUBE_URL_RE = re.compile(
    r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})"
)

SAMPLE_TRANSCRIPTS: list[dict[str, Any]] = [
    {
        "id": "nlp-keynote-sample",
        "title": "Modern NLP & Retrieval Systems Keynote (Sample)",
        "description": "Timestamped tech talk covering transformers, tokenization, NER, and RAG evaluation.",
        "timed_items": [
            {"start": 0.0, "duration": 22.0, "text": "[Music] Welcome everyone to the Global AI Summit in San Francisco! Today we are exploring how Natural Language Processing has evolved from classic TF-IDF pipelines to modern retrieval-augmented generation."},
            {"start": 22.0, "duration": 24.0, "text": "Many teams assume large language models replace classical NLP, but that is not true. Tokenization, lemmatization, and Named Entity Recognition remain essential for clean indexing and reliable metadata filtering."},
            {"start": 46.0, "duration": 25.0, "text": "When Google and OpenAI researchers benchmarked domain QA systems in 2025, uncleaned transcripts often caused frustrating hallucinations and poor retrieval precision."},
            {"start": 71.0, "duration": 26.0, "text": "However, combining spaCy linguistic preprocessing with vector search dramatically improved accuracy! Engineers felt confident because every extracted entity and sentiment shift could be audited."},
            {"start": 97.0, "duration": 25.0, "text": "Wait, what happens when a model encounters sarcasm or technical jargon in user feedback? Without negation handling and segment-level emotion tracking, analytics dashboards miss critical warning signs."},
            {"start": 122.0, "duration": 28.0, "text": "To wrap up, building transparent NLP systems is an exciting opportunity for developers worldwide. Start with clean evaluation metrics like ROUGE and F1, and iterate steadily with your users!"},
        ],
    },
    {
        "id": "product-review-sample",
        "title": "Developer Tooling Review & Critique (Sample)",
        "description": "Balanced transcript with positive highlights, concerns, and frustration segments.",
        "timed_items": [
            {"start": 0.0, "duration": 20.0, "text": "Hey developers, today we are testing the new CloudDeploy CLI released in Seattle last Tuesday. Honestly, the installation was super fast and intuitive."},
            {"start": 20.0, "duration": 22.0, "text": "The interactive terminal UI looks fantastic and the documentation is remarkably clear. I really enjoyed how easily it connected to GitHub Actions."},
            {"start": 42.0, "duration": 24.0, "text": "Unfortunately, once we pushed a larger microservice bundle, things became very slow and frustrating. Two builds crashed with vague timeout errors and zero helpful diagnostics."},
            {"start": 66.0, "duration": 24.0, "text": "I am worried about using version 1.0 in production right now because the rollback command failed during our staging test. Hopefully Microsoft or the maintainers patch this bug soon."},
            {"start": 90.0, "duration": 20.0, "text": "Overall, it is a promising tool with brilliant design, but wait for the stability update before migrating critical workloads."},
        ],
    },
]


class AskRequest(BaseModel):
    video_id: str = Field(..., description="The 11-character YouTube video ID or URL")
    question: str = Field(..., description="The question to ask about the video transcript")


class NLPInputRequest(BaseModel):
    video_id: Optional[str] = Field(None, description="11-character YouTube video ID or URL")
    text: Optional[str] = Field(None, description="Raw transcript or text to analyze")
    timed_items: Optional[list[dict[str, Any]]] = Field(
        None,
        description="Optional list of timestamped transcript items {text, start, duration}",
    )
    num_segments: int = Field(6, ge=1, le=20, description="Number of timeline segments")
    summary_sentences: int = Field(3, ge=1, le=10, description="Number of extractive summary sentences")
    max_token_samples: int = Field(100, ge=10, le=500, description="Max token rows returned")


# Simple in-memory cache for the current video's retriever
current_video_id = None
current_retriever = None


def extract_video_id(raw_value: str) -> str:
    """Extract an 11-char YouTube video ID from a URL or return the trimmed ID."""
    value = (raw_value or "").strip()
    match = _YOUTUBE_URL_RE.search(value)
    if match:
        return match.group(1)
    return value


def load_video_transcript_with_segments(video_id: str) -> tuple[str, list[dict[str, Any]]]:
    """Fetch the transcript and timestamped segments for a YouTube video."""
    if YouTubeTranscriptApi is None:
        raise HTTPException(
            status_code=503,
            detail="youtube-transcript-api is not installed. Install dependencies from requirement.txt or paste raw text instead.",
        )

    try:
        logger.info(f"Fetching transcript for video_id: {video_id}")
        youtube = YouTubeTranscriptApi()
        transcript = youtube.fetch(video_id)

        timed_items = []
        text_parts = []
        for item in transcript:
            item_text = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "")
            item_start = getattr(item, "start", None)
            if item_start is None and isinstance(item, dict):
                item_start = item.get("start", 0.0)
            item_duration = getattr(item, "duration", None)
            if item_duration is None and isinstance(item, dict):
                item_duration = item.get("duration", 0.0)

            if item_text:
                text_parts.append(item_text)
                timed_items.append(
                    {
                        "text": item_text,
                        "start": float(item_start or 0.0),
                        "duration": float(item_duration or 0.0),
                    }
                )

        full_text = " ".join(text_parts)
        return full_text, timed_items

    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Error fetching transcript for video {video_id}")
        raise HTTPException(
            status_code=404,
            detail="Transcript is disabled or unavailable for this video.",
        )


def load_video_transcript(video_id: str) -> str:
    """Fetch the transcript for a YouTube video."""
    full_text, _ = load_video_transcript_with_segments(video_id)
    return full_text


def _resolve_nlp_input(request: NLPInputRequest) -> tuple[str, list[dict[str, Any]] | None, str]:
    """Resolve input text and optional timed items from either raw text or YouTube video_id."""
    raw_text = (request.text or "").strip()
    if raw_text:
        return raw_text, request.timed_items, "text"

    raw_video_id = (request.video_id or "").strip()
    if raw_video_id:
        video_id = extract_video_id(raw_video_id)
        full_text, timed_items = load_video_transcript_with_segments(video_id)
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Transcript is empty.")
        return full_text, timed_items, f"youtube:{video_id}"

    raise HTTPException(
        status_code=400,
        detail="Provide either 'text' or 'video_id' in the request body.",
    )


@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the interactive NLP & YouTube RAG Web UI directly from FastAPI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"message": "YouTube NLP & RAG Backend is running. Visit /docs for API docs."}
    return FileResponse(str(index_path))


@app.get("/health")
def health_check():
    """Report service health and available NLP / RAG modules."""
    nlp = get_spacy_nlp()
    return {
        "status": "ok",
        "spacy_model": nlp.meta.get("name", "en_blank"),
        "spacy_pipes": nlp.pipe_names,
        "rag_dependencies_available": all(
            dep is not None for dep in (YouTubeTranscriptApi, Chroma, OllamaEmbeddings, ChatOllama)
        ),
    }


@app.get("/nlp/samples")
def list_sample_transcripts():
    """Return built-in sample transcripts for immediate offline testing in the UI."""
    samples = []
    for sample in SAMPLE_TRANSCRIPTS:
        full_text = " ".join(item["text"] for item in sample["timed_items"])
        samples.append(
            {
                "id": sample["id"],
                "title": sample["title"],
                "description": sample["description"],
                "text": full_text,
                "timed_items": sample["timed_items"],
            }
        )
    return {"samples": samples}


@app.post("/nlp/analyze")
def nlp_analyze(request: NLPInputRequest):
    """Run the full NLP pipeline: preprocessing, POS, NER, keywords, sentiment/emotion timeline, and extractive summary."""
    text, timed_items, source = _resolve_nlp_input(request)
    analyzer = get_analyzer()
    result = analyzer.analyze(
        text=text,
        timed_items=timed_items,
        num_segments=request.num_segments,
        summary_sentences=request.summary_sentences,
        max_token_samples=request.max_token_samples,
    )
    return {"source": source, **result}


@app.post("/nlp/preprocess")
def nlp_preprocess(request: NLPInputRequest):
    """Run text cleaning, tokenization, stemming, lemmatization, n-grams, and readability metrics."""
    text, _, source = _resolve_nlp_input(request)
    result = preprocess_text(text, max_token_samples=request.max_token_samples)
    return {"source": source, **result}


@app.post("/nlp/entities")
def nlp_entities(request: NLPInputRequest):
    """Run Part-of-Speech tagging, Named Entity Recognition, noun phrase extraction, and TF-IDF keywords."""
    text, _, source = _resolve_nlp_input(request)
    result = analyze_linguistics(text)
    return {"source": source, **result}


@app.post("/nlp/sentiment")
def nlp_sentiment(request: NLPInputRequest):
    """Run overall and segment-wise sentiment polarity and emotion analysis."""
    text, timed_items, source = _resolve_nlp_input(request)
    result = analyze_sentiment_and_emotions(
        text=text,
        timed_items=timed_items,
        num_segments=request.num_segments,
    )
    return {"source": source, **result}


@app.post("/ask")
def ask_question(request: AskRequest):
    global current_video_id, current_retriever

    video_id = extract_video_id(request.video_id)
    question = request.question.strip()

    if not video_id:
        raise HTTPException(status_code=400, detail="video_id cannot be empty.")
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    if Chroma is None or OllamaEmbeddings is None or ChatOllama is None:
        raise HTTPException(
            status_code=503,
            detail="RAG dependencies (langchain-chroma, langchain-ollama) are not installed. Install requirement.txt and ensure Ollama is running.",
        )

    # Rebuild the vectorstore and retriever if the video changed
    if current_video_id != video_id or current_retriever is None:
        logger.info(f"Setting up new vectorstore for video_id: {video_id} (previous: {current_video_id})")

        transcript_text = load_video_transcript(video_id)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        chunks = text_splitter.split_text(transcript_text)

        if not chunks:
            raise HTTPException(status_code=400, detail="Transcript is empty.")

        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")

            # Ephemeral database runs in memory when no persist_directory is specified
            vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
            )

            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "lambda_mult": 0.5,
                },
            )

            current_video_id = video_id
            current_retriever = retriever
            logger.info(f"Successfully cached vectorstore for video_id: {video_id}")

        except Exception as e:
            logger.error(f"Error initializing vectorstore/embeddings: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize RAG database. Please ensure Ollama is running and 'nomic-embed-text' model is installed.",
            )

    try:
        retrieved_docs = current_retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving context from RAG store.",
        )

    try:
        llm = ChatOllama(model="llama3.2", temperature=0)

        prompt = f"""You are a helpful assistant.

        Answer ONLY using the provided transcript context.

        If the answer is not present in the context, say:
        "I don't know based on the transcript"

        Context:
        {context}

        Question:
{question}

Answer:"""

        response = llm.invoke(prompt)
        answer = response.content
        return {"answer": answer}

    except Exception as e:
        logger.error(f"Error generating answer with llama3.2: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate answer. Please ensure Ollama is running and 'llama3.2' model is installed.",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
