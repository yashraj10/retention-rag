import json
import chromadb
import numpy as np
from config import CHROMA_COLLECTION, CHROMA_DB_DIR
from pathlib import Path

client = chromadb.PersistentClient(path=str(Path(__file__).parent / CHROMA_DB_DIR))
col = client.get_collection(CHROMA_COLLECTION)
data = col.get(include=["documents", "metadatas", "embeddings"])

export = {
    "ids": data["ids"],
    "documents": data["documents"],
    "metadatas": data["metadatas"],
    "embeddings": [list(map(float, e)) for e in data["embeddings"]],
}
with open("kb_export.json", "w") as f:
    json.dump(export, f)
print(f"✅ Exported {len(data['ids'])} chunks to kb_export.json")
