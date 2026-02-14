# 🎯 Retention Decision Twin

A **Retrieval-Augmented Generation (RAG)** system that acts as an AI-powered "decision twin" for CRM / Retention managers. Given a user cohort scenario, it retrieves relevant knowledge from a curated retention strategy knowledge base and recommends the optimal next action — grounded in evidence with cited sources.

## Why This Exists

Retention teams constantly face the same question: *"What should we do about this cohort?"* The answer depends on context — engagement patterns, lifecycle stage, budget, and what the research says actually works. This system automates that decision by:

1. **Retrieving** the most relevant retention strategies from a knowledge base of 15+ expert sources
2. **Generating** a structured, cited recommendation with trade-offs and risk analysis
3. **Evaluating** response quality systematically using an LLM-as-judge framework

## Architecture

```
┌──────────────┐     ┌────────────┐     ┌──────────────┐
│  15+ Sources │────▶│  Chunking  │────▶│   Embeddings │
│ (web + YT)   │     │ (1500 char │     │ (Gemini 768d)│
└──────────────┘     │  + overlap)│     └──────┬───────┘
                     └────────────┘            │
                                               ▼
                                      ┌────────────────┐
                                      │    ChromaDB     │
                                      │ (cosine HNSW)  │
                                      └───────┬────────┘
                                              │
                     ┌────────────────────────┘
                     │  top-k retrieval
                     ▼
┌──────────────┐     ┌────────────────────────────────────┐
│  User Query  │────▶│        Prompt Builder (v1/v2)      │
│  (scenario)  │     │  query + retrieved chunks + spec   │
└──────────────┘     └──────────────┬─────────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  Gemini 2.0     │
                           │  Flash          │
                           └────────┬────────┘
                                    │
                                    ▼
                     ┌──────────────────────────┐
                     │  Structured Recommendation│
                     │  (action + rationale +    │
                     │   citations + risks)      │
                     └──────────────────────────┘
```

## Key Features

- **ChromaDB vector store** with cosine similarity (HNSW index) — production-grade retrieval replacing naive NumPy search
- **15+ curated sources** from Amplitude, HubSpot, Intercom, Gainsight, and more
- **Two prompt strategies** compared head-to-head (simple vs. structured with constraints)
- **LLM-as-judge evaluation** scoring responses on 4 dimensions across 15 test scenarios
- **Streamlit UI** for interactive demo with real-time retrieval visualization
- **4-configuration A/B comparison**: RAG vs. No-RAG × Prompt v1 vs. v2

## Evaluation Framework

Each of 15 test queries is run through all 4 configurations and scored by a separate LLM judge:

| Dimension | What It Measures |
|---|---|
| **Relevance** | Does the recommendation match the specific scenario? |
| **Faithfulness** | Is the response grounded in retrieved context (no hallucination)? |
| **Citation Quality** | Are chunk IDs cited correctly and meaningfully? |
| **Actionability** | Can a CRM manager actually act on this recommendation? |

## Setup

```bash
# 1. Clone and install
git clone <your-repo-url>
cd decision-twin
pip install -r requirements.txt

# 2. Set your API key
export GEMINI_API_KEY="your-key-here"

# 3. Ingest the knowledge base into ChromaDB
python ingest.py

# 4. Run evaluation (~20 min, 60 LLM calls)
python evaluate.py

# 5. Launch the Streamlit app
streamlit run app.py
```

## Project Structure

```
decision-twin/
├── config.py           # Settings, models, eval queries, decision spec
├── sources.json        # Curated knowledge base URLs (web + YouTube)
├── ingest.py           # Scrape → chunk → embed → ChromaDB pipeline
├── rag.py              # Retrieval + generation engine
├── evaluate.py         # LLM-as-judge evaluation framework
├── app.py              # Streamlit UI
├── requirements.txt
├── eval_results.csv    # (generated) Raw evaluation scores
├── eval_report.md      # (generated) Evaluation summary report
└── chroma_db/          # (generated) Persistent vector store
```

## Tech Stack

- **LLM**: Google Gemini 2.0 Flash (generation + evaluation)
- **Embeddings**: Gemini text-embedding-004 (768 dimensions)
- **Vector Store**: ChromaDB with HNSW cosine index
- **UI**: Streamlit
- **Data Sources**: 15+ web articles + YouTube transcripts on user retention

## Resume Bullet

> Built a RAG-powered retention decision engine using Gemini, ChromaDB, and Streamlit. Evaluated 4 system configurations across 15 test scenarios using an LLM-as-judge framework, measuring relevance, faithfulness, citation quality, and actionability — demonstrating measurable improvement from structured prompting and retrieval augmentation.

## Possible Extensions

- Add more data sources (PDFs, internal docs, Slack exports)
- Fine-tune chunk size and overlap with retrieval metrics (MRR, Precision@k)
- Implement hybrid search (BM25 + semantic)
- Add user feedback loop to track recommendation quality over time
- Deploy via Docker + cloud hosting for a persistent demo
