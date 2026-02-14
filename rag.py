"""
rag.py - Retrieval-Augmented Generation engine for the Decision Twin.

Supports:
    - RAG mode (retrieve + generate)
    - No-RAG baseline (generate without context)
    - Two prompt versions (v1 = simple, v2 = structured with constraints)
"""
import time
from pathlib import Path

import chromadb
import google.generativeai as genai

from config import (
    GEMINI_API_KEY,
    GEN_MODEL,
    EMBED_MODEL,
    TOP_K,
    CHROMA_COLLECTION,
    CHROMA_DB_DIR,
    DECISION_TWIN_SPEC,
)

genai.configure(api_key=GEMINI_API_KEY)


# ──────────────────────────────────────────────
# ChromaDB connection
# ──────────────────────────────────────────────

def _get_collection():
    db_path = Path(__file__).parent / CHROMA_DB_DIR
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_collection(CHROMA_COLLECTION)


# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────

def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """
    Retrieve top-k chunks from ChromaDB by cosine similarity.
    Returns list of dicts with keys: chunk_id, source, ref, text, score.
    """
    collection = _get_collection()

    # Embed the query
    q_emb = genai.embed_content(model=EMBED_MODEL, content=query)["embedding"]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        # ChromaDB returns cosine *distance*; convert to similarity
        similarity = 1.0 - results["distances"][0][i]
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "ref": results["metadatas"][0][i]["ref"],
            "text": results["documents"][0][i],
            "score": round(similarity, 4),
        })
    return chunks


# ──────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────

def _build_prompt_v1(query: str, context: str = "") -> str:
    """Simple prompt - minimal structure."""
    actions = "\n".join(f"- {a}" for a in DECISION_TWIN_SPEC["actions"])
    ctx_block = f"\nCONTEXT:\n{context}\n" if context else ""
    return f"""You are a decision twin for: {DECISION_TWIN_SPEC["user"]}.
Decision: {DECISION_TWIN_SPEC["decision"]}

Possible actions:
{actions}

Use the context below to answer. Be concise.
{ctx_block}
User question:
{query}

Return:
1) Recommended Action
2) 3 bullet rationale
3) Cite evidence using chunk ids like [web_0_c2]
"""


def _build_prompt_v2(query: str, context: str = "") -> str:
    """Structured prompt - constraints, required citations, risks, missing info."""
    actions = "\n".join(f"- {a}" for a in DECISION_TWIN_SPEC["actions"])
    constraints = "\n".join(f"- {c}" for c in DECISION_TWIN_SPEC["constraints"])
    ctx_block = f"\nCONTEXT:\n{context}\n" if context else ""
    return f"""You are a decision twin for: {DECISION_TWIN_SPEC["user"]}.
Decision: {DECISION_TWIN_SPEC["decision"]}

Possible actions:
{actions}

Constraints:
{constraints}

Rules:
- Use ONLY the provided CONTEXT as evidence. If the context does not support a recommendation, choose "Do nothing".
- Cite at least 2 chunk ids in the rationale when you recommend a non-trivial action.
- Output must follow EXACTLY this format:

Recommended Action: <one action from the list>
Why (3 bullets, each must cite):
- ... [chunk_id]
- ... [chunk_id]
- ... [chunk_id]
Risks / Trade-offs (2 bullets, cite if possible):
- ... [chunk_id]
- ... [chunk_id]
Missing info (if any):
- ...
{ctx_block}
User question:
{query}
"""


# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────

def _generate(prompt: str) -> str:
    """Call Gemini generation with retry."""
    model = genai.GenerativeModel(GEN_MODEL)
    for attempt in range(6):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            wait = min(120, 2 ** (attempt + 2))
            if attempt < 5:
                print(f"  Rate limit. Waiting {wait}s... (attempt {attempt+1}/6)")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Generation failed after 6 attempts: {e}")


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def answer(
    query: str,
    use_rag: bool = True,
    prompt_version: str = "v2",
    k: int = TOP_K,
) -> dict:
    """
    Generate a decision twin recommendation.

    Args:
        query: The user's retention scenario / question.
        use_rag: If True, retrieve context from ChromaDB.
        prompt_version: "v1" (simple) or "v2" (structured).
        k: Number of chunks to retrieve (only used if use_rag=True).

    Returns:
        dict with keys: answer, query, prompt_version, use_rag, chunks, prompt
    """
    chunks = []
    context = ""

    if use_rag:
        chunks = retrieve(query, k=k)
        context_blocks = []
        for c in chunks:
            context_blocks.append(
                f"[{c['chunk_id']}] source={c['source']} ref={c['ref']}\n{c['text']}"
            )
        context = "\n\n".join(context_blocks)

    if prompt_version == "v1":
        prompt = _build_prompt_v1(query, context)
    else:
        prompt = _build_prompt_v2(query, context)

    answer_text = _generate(prompt)

    return {
        "answer": answer_text,
        "query": query,
        "prompt_version": prompt_version,
        "use_rag": use_rag,
        "chunks": chunks,
        "prompt": prompt,
    }


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    test_q = "A cohort has declining weekly engagement and a 10-day inactivity gap. Budget is limited. What should we do next?"

    print("=" * 60)
    print("RAG v2")
    print("=" * 60)
    result = answer(test_q, use_rag=True, prompt_version="v2")
    print(result["answer"])
    print("\nRetrieved chunks:")
    for c in result["chunks"]:
        print(f"  {c['chunk_id']}  score={c['score']}  ref={c['ref'][:50]}")
