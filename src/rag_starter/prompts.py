from __future__ import annotations

from .document import RetrievedChunk


DEFAULT_SYSTEM_PROMPT = (
    "Use the retrieved context to answer the user's question. "
    "If the answer is not supported by the context, say so clearly."
)


def build_context_block(matches: list[RetrievedChunk]) -> str:
    sections = []
    for i, match in enumerate(matches, start=1):
        sections.append(f"[Context {i}]\n{match.chunk.text}")
    return "\n\n".join(sections)


def build_rag_prompt(question: str, matches: list[RetrievedChunk]) -> str:
    context = build_context_block(matches)
    return (
        f"System:\n{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"User Question:\n{question}\n\n"
        "Answer:"
    )
