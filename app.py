"""
app.py - Streamlit UI for the Retention Decision Twin.

Run with:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from config import DECISION_TWIN_SPEC, TOP_K
from rag import answer, retrieve

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Retention Decision Twin",
    page_icon="🎯",
    layout="wide",
)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    use_rag = st.toggle("Enable RAG (retrieval)", value=True)
    prompt_version = st.radio(
        "Prompt version",
        options=["v2", "v1"],
        format_func=lambda x: "v2 - Structured (recommended)" if x == "v2" else "v1 - Simple",
    )
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=15, value=TOP_K)

    st.divider()
    st.caption("**Decision Twin Spec**")
    st.write(f"**Role:** {DECISION_TWIN_SPEC['user']}")
    st.write("**Actions:**")
    for a in DECISION_TWIN_SPEC["actions"]:
        st.write(f"  - {a}")

    st.divider()
    st.caption("Built with Gemini + ChromaDB + Streamlit")

    # Show eval report if it exists
    report_path = Path(__file__).parent / "eval_report.md"
    if report_path.exists():
        with st.expander("📊 Evaluation Report"):
            st.markdown(report_path.read_text())

# ──────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────
st.title("🎯 Retention Decision Twin")
st.markdown(
    "Describe a user cohort scenario and get an evidence-based retention recommendation."
)

# Example queries
EXAMPLES = [
    "A cohort has declining weekly engagement and a 10-day inactivity gap. Budget is limited. What should we do?",
    "New users drop off after their first session. Onboarding completion is only 30%.",
    "Power users who logged in daily now only come once a week. What do you recommend?",
    "Free tier users who hit usage limits — 70% churn. How should we intervene?",
    "We launched a new feature but adoption is 8% after 2 weeks. Users seem unaware of it.",
]

st.markdown("**Try an example:**")
cols = st.columns(3)
for i, example in enumerate(EXAMPLES[:3]):
    with cols[i]:
        if st.button(example[:60] + "...", key=f"ex_{i}", use_container_width=True):
            st.session_state["query"] = example

# Extra examples
with st.expander("More example scenarios"):
    for i, example in enumerate(EXAMPLES[3:], start=3):
        if st.button(example[:80] + "...", key=f"ex_{i}", use_container_width=True):
            st.session_state["query"] = example

# Query input
query = st.text_area(
    "Describe your cohort scenario:",
    value=st.session_state.get("query", ""),
    height=100,
    placeholder="e.g., Users signed up during a promo campaign but never used the core feature. Churn is high...",
)

if st.button("🔍 Get Recommendation", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a scenario.")
    else:
        with st.spinner("Retrieving context and generating recommendation..."):
            result = answer(
                query=query.strip(),
                use_rag=use_rag,
                prompt_version=prompt_version,
                k=k,
            )

        # -- Recommendation --
        st.divider()
        st.subheader("📋 Recommendation")
        st.markdown(result["answer"])

        # -- Retrieved chunks --
        if result["chunks"]:
            st.divider()
            st.subheader("📚 Retrieved Context")

            chunks_df = pd.DataFrame(result["chunks"])
            chunks_df = chunks_df[["chunk_id", "source", "ref", "score"]]
            st.dataframe(
                chunks_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "Similarity",
                        min_value=0,
                        max_value=1,
                        format="%.3f",
                    ),
                    "ref": st.column_config.LinkColumn("Source URL"),
                },
            )

            # Expandable full text
            with st.expander("View full chunk text"):
                for c in result["chunks"]:
                    st.markdown(f"**{c['chunk_id']}** (score: {c['score']})")
                    st.text(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))
                    st.divider()

        # -- Debug: full prompt --
        with st.expander("🔧 Debug: Full prompt sent"):
            st.code(result["prompt"], language="text")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Retention Decision Twin - RAG-powered recommendation engine | "
    f"Config: {'RAG' if use_rag else 'No-RAG'} + {prompt_version}"
)
