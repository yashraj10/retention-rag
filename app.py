"""Streamlit UI — Retention Decision Twin."""

import streamlit as st
import json
import chromadb
import tempfile
import os
from pathlib import Path

from config import CHROMA_COLLECTION, TOP_K
from rag import answer

# ─────────────────────────────────────────
# KB loader (JSON → writable tmp ChromaDB)
# ─────────────────────────────────────────
@st.cache_resource
def ensure_db():
    tmp_dir = os.path.join(tempfile.gettempdir(), "chroma_retention")
    client = chromadb.PersistentClient(path=tmp_dir)
    try:
        col = client.get_collection(CHROMA_COLLECTION)
        if col.count() > 0:
            return tmp_dir
    except Exception:
        pass
    json_path = Path(__file__).parent / "kb_export.json"
    with open(json_path) as f:
        data = json.load(f)
    col = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    batch = 100
    for i in range(0, len(data["ids"]), batch):
        col.upsert(
            ids=data["ids"][i:i+batch],
            documents=data["documents"][i:i+batch],
            metadatas=data["metadatas"][i:i+batch],
            embeddings=data["embeddings"][i:i+batch],
        )
    return tmp_dir

db_path = ensure_db()

import config
config.CHROMA_DB_DIR = db_path
import rag
rag.CHROMA_DB_DIR = db_path


# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Retention Decision Twin",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
# Custom CSS — Minimal Light Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #ffffff;
        --bg-subtle: #f8f9fa;
        --bg-muted: #f1f3f5;
        --border: #e9ecef;
        --border-strong: #dee2e6;
        --text: #1a1a1a;
        --text-secondary: #495057;
        --text-muted: #868e96;
        --text-faint: #adb5bd;
        --accent: #1a1a1a;
        --accent-hover: #343a40;
        --green: #2b8a3e;
        --green-bg: #ebfbee;
        --tag-bg: #f1f3f5;
        --tag-border: #dee2e6;
    }

    /* ── Global ── */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .stApp { color: var(--text) !important; }

    header[data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
    .block-container { padding: 2.5rem 3rem 3rem !important; max-width: 960px !important; }

    h1, h2, h3, h4, p, li, span, div, label, .stMarkdown {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
    }

    /* ── Top Bar ── */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 2rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border);
    }
    .topbar-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .topbar-logo {
        width: 28px; height: 28px;
        border-radius: 7px;
        background: var(--accent);
        color: #fff;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; font-weight: 600;
    }
    .topbar-name {
        font-size: 0.9rem; font-weight: 600;
        color: var(--text) !important;
        letter-spacing: -0.01em;
    }
    .topbar-tags { display: flex; gap: 6px; }
    .topbar-tag {
        padding: 3px 10px;
        background: var(--tag-bg);
        border: 1px solid var(--tag-border);
        border-radius: 5px;
        font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        color: var(--text-muted) !important;
    }

    /* ── Hero ── */
    .hero-title {
        font-size: 1.65rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em;
        line-height: 1.25 !important;
        margin: 0 0 8px !important;
        color: var(--text) !important;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: var(--text-muted) !important;
        line-height: 1.6;
        max-width: 560px;
        margin-bottom: 2rem;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-faint) !important;
        margin-bottom: 10px;
    }

    /* ── Scenario Buttons ── */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: var(--bg) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 400 !important;
        font-size: 0.82rem !important;
        padding: 14px 16px !important;
        text-align: left !important;
        line-height: 1.5 !important;
        transition: all 0.15s ease !important;
        min-height: 90px !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        border-color: var(--border-strong) !important;
        background: var(--bg-subtle) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button p {
        color: var(--text-secondary) !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button strong {
        color: var(--text) !important; font-weight: 600 !important;
    }

    /* ── Input ── */
    .stTextArea textarea {
        background: var(--bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 14px !important;
        line-height: 1.6 !important;
        transition: border-color 0.15s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }
    .stTextArea textarea::placeholder { color: var(--text-faint) !important; }
    .stTextArea label { display: none !important; }

    /* ── Primary Button ── */
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 7px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.55rem 1.4rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover) !important;
    }

    /* ── Result ── */
    .result-card {
        background: var(--bg-subtle);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 24px 28px 8px;
        margin-top: 1.5rem;
    }
    .result-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 16px;
        padding-bottom: 14px;
        border-bottom: 1px solid var(--border);
    }
    .result-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--green);
    }
    .result-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--green) !important;
    }

    /* ── Sources ── */
    .streamlit-expanderHeader {
        background: var(--bg-subtle) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        color: var(--text-muted) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    .source-row {
        display: flex; align-items: baseline; gap: 10px;
        padding: 6px 0; font-size: 0.8rem;
    }
    .source-id {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.73rem;
        color: var(--text-muted) !important;
        background: var(--tag-bg);
        padding: 2px 8px; border-radius: 4px;
        white-space: nowrap;
    }
    .source-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--green) !important;
        font-weight: 500;
    }
    .source-domain {
        font-size: 0.76rem;
        color: var(--text-faint) !important;
    }

    /* ── Footer ── */
    .app-footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
        font-size: 0.72rem;
        color: var(--text-faint) !important;
        display: flex; align-items: center; justify-content: space-between;
    }
    .footer-tags { display: flex; gap: 6px; }
    .footer-tag {
        padding: 2px 8px;
        background: var(--tag-bg);
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-faint) !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-subtle) !important;
        border-right: 1px solid var(--border) !important;
    }

    hr { border-color: var(--border) !important; opacity: 0.6; }
    .stSpinner > div { color: var(--text-muted) !important; }
    [data-testid="stExpander"] { border: none !important; }
    .stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Session state
# ─────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""
if "result" not in st.session_state:
    st.session_state.result = None


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    use_rag = st.toggle("RAG retrieval", value=True)
    prompt_version = st.radio(
        "Prompt version",
        ["v2", "v1"],
        format_func=lambda v: "v2 — Structured" if v == "v2" else "v1 — Simple",
    )
    k = st.slider("Chunks to retrieve", 1, 10, TOP_K)
    st.markdown("---")
    st.caption("Knowledge base: 123 chunks from 6 curated retention strategy articles.")


# ─────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────
rag_label = "RAG on" if use_rag else "RAG off"
st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-logo">R</div>
        <span class="topbar-name">Retention Decision Twin</span>
    </div>
    <div class="topbar-tags">
        <span class="topbar-tag">{rag_label}</span>
        <span class="topbar-tag">Prompt {prompt_version}</span>
        <span class="topbar-tag">k={k}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Hero
# ─────────────────────────────────────────
st.markdown('<h1 class="hero-title">What retention action should you take?</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Describe a user cohort scenario. The engine retrieves evidence from curated retention research and generates a structured recommendation.</p>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────
# Example Scenarios
# ─────────────────────────────────────────
examples = [
    {"label": "Engagement drop-off", "text": "A cohort has declining weekly engagement and a 10-day inactivity gap. Budget is limited. What should we do?"},
    {"label": "Onboarding failure", "text": "New users drop off after their first session. Onboarding completion is only 30%. How do we improve retention?"},
    {"label": "Power user decline", "text": "Power users who logged in daily now only come once a week. What's the best re-engagement strategy?"},
    {"label": "Free tier churn", "text": "Free tier users who hit usage limits — 70% churn. How should we intervene before they hit the wall?"},
    {"label": "Low feature adoption", "text": "We launched a new feature but adoption is 8% after 2 weeks. Users seem unaware of it. What should we do?"},
    {"label": "Promo cohort churn", "text": "Users who signed up during a promo campaign never used the core feature. Churn is high. What should we do?"},
]

st.markdown('<div class="section-label">Example scenarios</div>', unsafe_allow_html=True)

cols = st.columns(3)
for i, ex in enumerate(examples[:6]):
    with cols[i % 3]:
        if st.button(f"**{ex['label']}**\n\n{ex['text'][:75]}...", key=f"ex_{i}", use_container_width=True):
            st.session_state.query = ex["text"]
            st.session_state.result = None
            st.rerun()


# ─────────────────────────────────────────
# Input
# ─────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1.8rem;">Your scenario</div>', unsafe_allow_html=True)

query = st.text_area(
    "scenario",
    value=st.session_state.query,
    height=110,
    placeholder="e.g., Users signed up during a promo campaign but never activated the core feature. 60-day churn rate is 45%...",
    label_visibility="collapsed",
)

run = st.button("Generate recommendation", type="primary")


# ─────────────────────────────────────────
# Generate
# ─────────────────────────────────────────
if run:
    if not query.strip():
        st.warning("Enter a scenario first.")
    else:
        with st.spinner("Retrieving context and generating recommendation..."):
            result = answer(
                query=query.strip(),
                use_rag=use_rag,
                prompt_version=prompt_version,
                k=k,
            )
            st.session_state.result = result
            st.session_state.query = query


# ─────────────────────────────────────────
# Display Result
# ─────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result

    st.markdown("""
    <div class="result-card">
        <div class="result-header">
            <div class="result-dot"></div>
            <span class="result-label">Recommendation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(result["answer"])

    if result.get("chunks"):
        with st.expander(f"Retrieved sources  ·  {len(result['chunks'])} chunks"):
            for c in result["chunks"]:
                domain = c["ref"].split("/")[2] if "://" in c["ref"] else c["ref"][:40]
                st.markdown(
                    f'<div class="source-row">'
                    f'<span class="source-id">{c["chunk_id"]}</span>'
                    f'<span class="source-score">{c["score"]:.3f}</span>'
                    f'<span class="source-domain">{domain}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(c["text"][:200] + "...")
                st.markdown("---")


# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.markdown(f"""
<div class="app-footer">
    <span>Retention Decision Twin</span>
    <div class="footer-tags">
        <span class="footer-tag">Gemini 2.0 Flash</span>
        <span class="footer-tag">ChromaDB</span>
        <span class="footer-tag">123 chunks</span>
    </div>
</div>
""", unsafe_allow_html=True)