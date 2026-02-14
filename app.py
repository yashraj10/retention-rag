"""Streamlit UI — Retention Decision Twin (Professional Edition)."""

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
# Custom CSS — Professional Dark Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0a0a0b;
        --bg-secondary: #111113;
        --bg-card: #16161a;
        --bg-card-hover: #1c1c21;
        --border: #232328;
        --border-accent: #2a2a32;
        --text-primary: #ececef;
        --text-secondary: #8b8b96;
        --text-muted: #5a5a66;
        --accent: #6c5ce7;
        --accent-dim: #6c5ce720;
        --accent-glow: #6c5ce730;
        --success: #00b894;
        --success-dim: #00b89415;
        --warning: #fdcb6e;
        --tag-bg: #1e1e24;
    }

    /* ── Global ── */
    .stApp, [data-testid="stAppViewContainer"], section[data-testid="stSidebar"] {
        background-color: var(--bg-primary) !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }

    .stApp { color: var(--text-primary) !important; }

    /* Kill default Streamlit chrome */
    header[data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
    .block-container { padding: 2rem 3rem 3rem !important; max-width: 1100px !important; }

    /* ── Typography ── */
    h1, h2, h3, h4, p, li, span, div, label, .stMarkdown {
        font-family: 'DM Sans', -apple-system, sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* ── Hero Section ── */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        background: var(--accent-dim);
        border: 1px solid var(--accent);
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--accent) !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 16px;
    }
    .hero-title {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em;
        line-height: 1.15 !important;
        margin: 0 0 10px !important;
        color: var(--text-primary) !important;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: var(--text-secondary) !important;
        line-height: 1.6;
        max-width: 620px;
        margin-bottom: 2rem;
    }

    /* ── Scenario Cards ── */
    .scenario-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
        gap: 12px;
        margin: 1.2rem 0 2rem;
    }
    .scenario-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 18px;
        cursor: pointer;
        transition: all 0.2s ease;
        position: relative;
    }
    .scenario-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--accent);
        box-shadow: 0 0 20px var(--accent-glow);
        transform: translateY(-1px);
    }
    .scenario-card .sc-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--accent) !important;
        margin-bottom: 6px;
    }
    .scenario-card .sc-text {
        font-size: 0.88rem;
        line-height: 1.5;
        color: var(--text-secondary) !important;
    }

    /* ── Input Area ── */
    .stTextArea textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 16px !important;
        transition: border-color 0.2s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-dim) !important;
    }
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* ── Primary Button ── */
    .stButton > button[kind="primary"], .stButton > button {
        background: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.65rem 1.6rem !important;
        letter-spacing: 0.01em;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #5a4bd4 !important;
        box-shadow: 0 4px 20px var(--accent-glow) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Result Card ── */
    .result-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 28px 30px;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .result-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent), var(--success), var(--accent));
        background-size: 200% 100%;
        animation: shimmer 3s ease infinite;
    }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    .result-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 18px;
    }
    .result-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: var(--success-dim);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .result-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--success) !important;
    }
    .result-body {
        font-size: 0.92rem;
        line-height: 1.75;
        color: var(--text-secondary) !important;
    }
    .result-body strong, .result-body b {
        color: var(--text-primary) !important;
        font-weight: 600;
    }

    /* ── Sources Expander ── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        color: var(--text-secondary) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* ── Source Chip ── */
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        background: var(--tag-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        font-size: 0.76rem;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-muted) !important;
        margin: 3px 4px 3px 0;
    }
    .source-chip .sc-score {
        color: var(--success) !important;
        font-weight: 500;
    }

    /* ── Config Bar ── */
    .config-bar {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 12px 18px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    .config-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        color: var(--text-muted) !important;
    }
    .config-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--success);
    }
    .config-dot.off { background: var(--text-muted); }

    /* ── Footer ── */
    .app-footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }
    .footer-left {
        font-size: 0.75rem;
        color: var(--text-muted) !important;
    }
    .footer-tags {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }
    .footer-tag {
        padding: 3px 10px;
        background: var(--tag-bg);
        border: 1px solid var(--border);
        border-radius: 6px;
        font-size: 0.68rem;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-muted) !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stCheckbox label {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }

    /* ── Toggle / Radio overrides ── */
    .stRadio > div { gap: 4px !important; }
    .stRadio > div > label {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        font-size: 0.82rem !important;
    }

    /* ── Section Label ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted) !important;
        margin-bottom: 10px;
    }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; opacity: 0.5; }

    /* ── Spinner ── */
    .stSpinner > div { color: var(--text-secondary) !important; }

    /* ── Hide Streamlit label "press enter to apply" etc ── */
    .stTextArea label { display: none !important; }

    /* ── Make Streamlit elements blend ── */
    [data-testid="stExpander"] { border: none !important; }
    .stAlert { border-radius: 10px !important; }
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
# Sidebar — Settings
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
    st.markdown(
        '<p style="font-size:0.75rem;color:#5a5a66;">Knowledge base: 123 chunks from 6 curated retention sources.</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
# Hero
# ─────────────────────────────────────────
st.markdown('<div class="hero-badge">◎ RAG-Powered Decision Engine</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Retention Decision Twin</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Describe a user cohort scenario. Get an evidence-based retention recommendation grounded in curated strategy research.</p>',
    unsafe_allow_html=True,
)

# Config bar
rag_dot = "config-dot" if use_rag else "config-dot off"
st.markdown(f"""
<div class="config-bar">
    <div class="config-item">
        <div class="{rag_dot}"></div>
        RAG {'enabled' if use_rag else 'disabled'}
    </div>
    <div class="config-item">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#5a5a66" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
        Prompt {prompt_version}
    </div>
    <div class="config-item">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#5a5a66" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        Top-{k} chunks
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Example Scenarios
# ─────────────────────────────────────────
examples = [
    {
        "label": "Engagement Drop",
        "text": "A cohort has declining weekly engagement and a 10-day inactivity gap. Budget is limited. What should we do?",
    },
    {
        "label": "Onboarding Failure",
        "text": "New users drop off after their first session. Onboarding completion is only 30%. How do we improve retention?",
    },
    {
        "label": "Power User Decline",
        "text": "Power users who logged in daily now only come once a week. What's the best re-engagement strategy?",
    },
    {
        "label": "Free Tier Churn",
        "text": "Free tier users who hit usage limits — 70% churn. How should we intervene before they hit the wall?",
    },
    {
        "label": "Feature Adoption",
        "text": "We launched a new feature but adoption is 8% after 2 weeks. Users seem unaware of it. What should we do?",
    },
    {
        "label": "Promo Cohort Churn",
        "text": "Users who signed up during a promo campaign never used the core feature. Churn is high. What should we do?",
    },
]

st.markdown('<div class="section-label">Example scenarios</div>', unsafe_allow_html=True)

cols = st.columns(3)
for i, ex in enumerate(examples[:6]):
    col = cols[i % 3]
    with col:
        if st.button(
            f"**{ex['label']}**\n\n{ex['text'][:80]}...",
            key=f"ex_{i}",
            use_container_width=True,
        ):
            st.session_state.query = ex["text"]
            st.session_state.result = None
            st.rerun()


# ─────────────────────────────────────────
# Input
# ─────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1.5rem;">Describe your scenario</div>', unsafe_allow_html=True)

query = st.text_area(
    "scenario",
    value=st.session_state.query,
    height=120,
    placeholder="e.g., Users signed up during a promo campaign but never activated the core feature. 60-day churn rate is 45%. Budget for incentives is limited...",
    label_visibility="collapsed",
)

col_btn, col_space = st.columns([1, 3])
with col_btn:
    run = st.button("Generate recommendation", type="primary", use_container_width=True)


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

    # Format the answer text with better markdown
    answer_text = result["answer"].replace("**Recommended Action:**", "### Recommended Action\n").replace("Recommended Action:", "### Recommended Action\n")

    st.markdown(f"""
    <div class="result-container">
        <div class="result-header">
            <div class="result-icon">✦</div>
            <div>
                <div class="result-label">AI Recommendation</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Render markdown inside streamlit for proper formatting
    st.markdown(answer_text)

    # Sources
    if result.get("chunks"):
        with st.expander(f"View retrieved sources ({len(result['chunks'])} chunks)"):
            for c in result["chunks"]:
                domain = c["ref"].split("/")[2] if "://" in c["ref"] else c["ref"][:40]
                st.markdown(
                    f'<div class="source-chip">'
                    f'<span class="sc-score">{c["score"]:.3f}</span> '
                    f'{c["chunk_id"]} · {domain}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(c["text"][:250] + "...")
                st.markdown("---")


# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.markdown(f"""
<div class="app-footer">
    <div class="footer-left">Retention Decision Twin · RAG-powered recommendation engine</div>
    <div class="footer-tags">
        <span class="footer-tag">Gemini 2.0 Flash</span>
        <span class="footer-tag">ChromaDB</span>
        <span class="footer-tag">123 chunks</span>
        <span class="footer-tag">{'RAG + ' + prompt_version if use_rag else 'No-RAG + ' + prompt_version}</span>
    </div>
</div>
""", unsafe_allow_html=True)