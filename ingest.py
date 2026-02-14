"""
ingest.py - Scrape sources, chunk, embed, and store in ChromaDB.

Usage:
    python ingest.py                # ingest all sources from sources.json
    python ingest.py --reset        # wipe DB and re-ingest
"""
import argparse
import json
import re
import time
import shutil
from pathlib import Path

import chromadb
import google.generativeai as genai
import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import (
    GEMINI_API_KEY,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_COLLECTION,
    CHROMA_DB_DIR,
)

genai.configure(api_key=GEMINI_API_KEY)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def clean_text(s: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", s).strip()


def fetch_web_text(url: str) -> str:
    """Scrape readable text from a URL."""
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return clean_text(soup.get_text(separator=" "))


def fetch_youtube_transcript(video_id: str) -> str:
    """Fetch transcript for a YouTube video."""
    from youtube_transcript_api import YouTubeTranscriptApi
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([x["text"] for x in transcript])
    return clean_text(text)


# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ──────────────────────────────────────────────
# Embedding (with rate-limit retry)
# ──────────────────────────────────────────────

def embed_batch(texts: list[str], batch_size: int = 10) -> list[list[float]]:
    """Embed texts with Gemini, handling rate limits gracefully."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        retries = 0
        while retries < 8:
            try:
                results = genai.embed_content(
                    model=EMBED_MODEL,
                    content=batch,
                )
                all_embeddings.extend(results["embedding"])
                break
            except Exception as e:
                retries += 1
                wait = min(60, 2 ** retries)
                print(f"  Rate limit hit. Waiting {wait}s... (attempt {retries}/8)")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to embed batch starting at index {i}")
        # Pause between batches to stay under 100 req/min
        time.sleep(8)
    return all_embeddings


# ──────────────────────────────────────────────
# Main ingestion pipeline
# ──────────────────────────────────────────────

def ingest(reset: bool = False):
    sources_path = Path(__file__).parent / "sources.json"
    with open(sources_path) as f:
        sources = json.load(f)

    # -- 1. Fetch raw documents --
    docs = []
    print("\n📥  Fetching web sources...")
    for i, url in enumerate(sources.get("web_urls", [])):
        try:
            txt = fetch_web_text(url)
            docs.append({"doc_id": f"web_{i}", "source": "web", "ref": url, "text": txt})
            print(f"  ✓ web_{i}: {url[:60]}...  ({len(txt):,} chars)")
        except Exception as e:
            print(f"  ✗ web_{i}: {url[:60]}...  -> {e}")

    print("\n📥  Fetching YouTube transcripts...")
    for j, vid in enumerate(sources.get("youtube_video_ids", [])):
        try:
            txt = fetch_youtube_transcript(vid)
            docs.append({
                "doc_id": f"yt_{j}",
                "source": "youtube",
                "ref": f"https://youtube.com/watch?v={vid}",
                "text": txt,
            })
            print(f"  ✓ yt_{j}: {vid}  ({len(txt):,} chars)")
        except Exception as e:
            print(f"  ✗ yt_{j}: {vid}  -> {e}")

    if not docs:
        print("❌  No documents fetched. Check your URLs and network.")
        return

    # -- 2. Chunk --
    print(f"\n✂️  Chunking {len(docs)} documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    all_chunks = []
    for doc in docs:
        for k, chunk_text_ in enumerate(chunk_text(doc["text"])):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_c{k}",
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "ref": doc["ref"],
                "text": chunk_text_,
            })
    print(f"  -> {len(all_chunks)} chunks total")

    # -- 3. Embed --
    print(f"\n🔢  Embedding {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_batch(texts)

    # -- 4. Store in ChromaDB --
    db_path = Path(__file__).parent / CHROMA_DB_DIR
    if reset and db_path.exists():
        print("🗑️  Resetting existing ChromaDB...")
        shutil.rmtree(db_path)

    print(f"\n💾  Storing in ChromaDB ({CHROMA_DB_DIR})...")
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches (ChromaDB handles dedup via IDs)
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings[i : i + batch_size],
            documents=[c["text"] for c in batch],
            metadatas=[
                {"doc_id": c["doc_id"], "source": c["source"], "ref": c["ref"]}
                for c in batch
            ],
        )

    print(f"\n✅  Done! {collection.count()} chunks in collection '{CHROMA_COLLECTION}'")
    print(f"   Sources ingested: {len(docs)} documents -> {len(all_chunks)} chunks")


# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest retention KB into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Wipe DB and re-ingest")
    args = parser.parse_args()
    ingest(reset=args.reset)
