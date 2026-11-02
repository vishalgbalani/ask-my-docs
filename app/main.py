"""FastAPI app — Ask My Docs."""

import os
import uuid
import time
import threading
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

from .chunker import Chunk, chunk_pdf, chunk_docx, chunk_plain_text
from .embeddings import get_embeddings, retrieve_chunks, get_confidence
from .answerer import generate_answer

load_dotenv()

# ── Session storage ──────────────────────────────────────────────
# Each session: { chunks, embeddings, document_info, conversation_history, last_access, questions_asked }
sessions: dict = {}
SESSION_TTL = 1800  # 30 minutes
MAX_QUESTIONS = 10


def cleanup_sessions():
    """Remove expired sessions."""
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s["last_access"] > SESSION_TTL]
    for sid in expired:
        del sessions[sid]


def start_cleanup_timer():
    """Run cleanup every 5 minutes."""
    cleanup_sessions()
    timer = threading.Timer(300, start_cleanup_timer)
    timer.daemon = True
    timer.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_cleanup_timer()
    yield


app = FastAPI(title="Ask My Docs", lifespan=lifespan)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    return OpenAI(api_key=api_key)


# ── Models ───────────────────────────────────────────────────────

class IndexTextRequest(BaseModel):
    text: str
    title: Optional[str] = "Pasted Document"


class AskRequest(BaseModel):
    session_id: str
    question: str


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
    return FileResponse(html_path)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a PDF or DOCX document."""
    filename_lower = file.filename.lower()
    if not (filename_lower.endswith(".pdf") or filename_lower.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Please upload a PDF or DOCX file.")

    contents = await file.read()

    # 10MB limit
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File is too large. Maximum size is 10MB.")

    page_count = None

    try:
        if filename_lower.endswith(".pdf"):
            chunks, page_count = chunk_pdf(contents)
        else:
            chunks = chunk_docx(contents)
    except ValueError as e:
        if str(e) == "no_extractable_text":
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from this file. For PDFs, scanned documents require OCR which is not supported yet."
            )
        raise HTTPException(status_code=400, detail=str(e))

    if not chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from this file.")

    # Generate embeddings
    client = get_client()
    texts = [c.text for c in chunks]
    embeddings = get_embeddings(texts, client)

    # Create session
    session_id = str(uuid.uuid4())
    total_chars = sum(len(c.text) for c in chunks)

    sessions[session_id] = {
        "chunks": chunks,
        "embeddings": embeddings,
        "document_info": {
            "filename": file.filename,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "total_characters": total_chars
        },
        "conversation_history": [],
        "last_access": time.time(),
        "questions_asked": 0
    }

    return {
        "session_id": session_id,
        "document": sessions[session_id]["document_info"]
    }


@app.post("/index-text")
async def index_text(request: IndexTextRequest):
    """Index pasted text."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Please paste some text to index.")

    try:
        chunks = chunk_plain_text(request.text, request.title)
    except ValueError as e:
        if str(e) == "text_too_short":
            raise HTTPException(
                status_code=400,
                detail="The text is too short to index meaningfully. Paste at least a few paragraphs."
            )
        raise HTTPException(status_code=400, detail=str(e))

    # Generate embeddings
    client = get_client()
    texts = [c.text for c in chunks]
    embeddings = get_embeddings(texts, client)

    # Create session
    session_id = str(uuid.uuid4())
    total_chars = sum(len(c.text) for c in chunks)

    sessions[session_id] = {
        "chunks": chunks,
        "embeddings": embeddings,
        "document_info": {
            "filename": request.title,
            "page_count": None,
            "chunk_count": len(chunks),
            "total_characters": total_chars
        },
        "conversation_history": [],
        "last_access": time.time(),
        "questions_asked": 0
    }

    return {
        "session_id": session_id,
        "document": sessions[session_id]["document_info"]
    }


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Ask a question about the indexed document."""
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    session["last_access"] = time.time()

    # Rate limit
    if session["questions_asked"] >= MAX_QUESTIONS:
        raise HTTPException(
            status_code=429,
            detail="Question limit reached for this session. Upload a new document to start a new session."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")

    client = get_client()

    # Embed the question
    query_embedding = get_embeddings([request.question], client)[0]

    # Retrieve relevant chunks
    retrieved = retrieve_chunks(
        query_embedding,
        session["embeddings"],
        session["chunks"],
        top_k=5
    )

    # Check if any chunks are relevant
    top_score = max(score for _, score in retrieved) if retrieved else 0
    confidence = get_confidence(top_score)

    # Generate answer
    answer = generate_answer(
        question=request.question,
        retrieved=retrieved,
        confidence=confidence,
        client=client,
        conversation_history=session["conversation_history"]
    )

    # Update session
    session["questions_asked"] += 1
    session["conversation_history"].append({
        "question": request.question,
        "answer": answer.answer
    })

    return answer.model_dump()
