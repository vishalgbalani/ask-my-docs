"""Answer generation using retrieved chunks."""

from typing import List
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

from .chunker import Chunk

ANSWER_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a research assistant for Product Managers. You answer questions about documents using ONLY the provided source excerpts.

Rules:
1. Answer based ONLY on the provided excerpts. Do not use general knowledge.
2. Look carefully for information that is RELEVANT to the question, even if the document uses different terminology. For example, "risks" may appear as "pain points", "key considerations", "challenges", "blockers", or "dependencies". "Assumptions" may appear as "constraints", "inputs", "parameters", or "baseline values".
3. If after careful review you truly find NOTHING relevant in any excerpt, say: "I couldn't find this in the document. Try rephrasing your question."
4. Cite your sources using [Section X] tags, where X is the section number shown.
5. Be specific and concise. PMs want direct answers, not summaries of what the document says about the topic.
6. If the question asks for a summary, provide bullet points with citations.
7. If the question asks for risks, decisions, or recommendations, extract them directly from the text — look for implicit risks and decisions too, not just explicitly labeled ones.
8. Preserve the document's own language for key terms and definitions — don't paraphrase technical terms.
9. Keep answers to 2-4 paragraphs unless a longer response is clearly needed.
10. When in doubt, attempt an answer from the available excerpts rather than refusing. A partial answer with citations is more useful than no answer."""


class Citation(BaseModel):
    chunk_id: int
    page_number: Optional[int] = None
    text_preview: str = Field(description="First 100 chars of the chunk")


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    chunks_retrieved: int
    confidence: str


def build_context(retrieved: List[tuple[Chunk, float]]) -> str:
    """Build context string from retrieved chunks."""
    parts = []
    for chunk, score in retrieved:
        page_info = f" (Page {chunk.page_number})" if chunk.page_number else ""
        parts.append(f"[Section {chunk.chunk_id}]{page_info}:\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    retrieved: List[tuple[Chunk, float]],
    confidence: str,
    client: OpenAI,
    conversation_history: List[dict] = None
) -> AnswerResponse:
    """Generate an answer from retrieved chunks."""

    context = build_context(retrieved)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add conversation history for follow-ups
    if conversation_history:
        for entry in conversation_history[-4:]:  # last 4 Q&A pairs max
            messages.append({"role": "user", "content": entry["question"]})
            messages.append({"role": "assistant", "content": entry["answer"]})

    user_message = f"""Source excerpts from the document:

{context}

---

Question: {question}"""

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1500
    )

    answer_text = response.choices[0].message.content

    # Build citations from retrieved chunks
    citations = []
    for chunk, score in retrieved:
        citations.append(Citation(
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            text_preview=chunk.text[:100]
        ))

    return AnswerResponse(
        question=question,
        answer=answer_text,
        citations=citations,
        chunks_retrieved=len(retrieved),
        confidence=confidence
    )
