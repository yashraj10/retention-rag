"""
Configuration for the Decision Twin RAG system.
"""
import os

# ──────────────────────────────────────────────
# API  (set via env var or edit here)
# ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

GEN_MODEL = "models/gemini-2.0-flash"          # generation
EVAL_MODEL = "models/gemini-2.0-flash"          # LLM-as-judge
EMBED_MODEL = "models/gemini-embedding-001"       # embeddings (768-d)

# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────
CHUNK_SIZE = 1500       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between consecutive chunks

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
TOP_K = 5               # number of chunks to retrieve
CHROMA_COLLECTION = "retention_kb"
CHROMA_DB_DIR = "chroma_db"

# ──────────────────────────────────────────────
# Decision Twin specification
# ──────────────────────────────────────────────
DECISION_TWIN_SPEC = {
    "user": "Retention / CRM Manager",
    "decision": "Recommend the next best retention action for a user cohort",
    "actions": [
        "Do nothing",
        "Send educational reminder",
        "Send personalized highlight notification",
        "Offer limited-time incentive",
        "Escalate to human support",
    ],
    "constraints": [
        "Must cite evidence from retrieved context",
        "If evidence is insufficient, recommend 'Do nothing' and explain what data is missing",
        "Avoid discriminatory recommendations based on protected attributes",
    ],
}

# ──────────────────────────────────────────────
# Evaluation test queries (15 diverse scenarios)
# ──────────────────────────────────────────────
EVAL_QUERIES = [
    "A cohort has declining weekly engagement and a 10-day inactivity gap. Budget is limited. What should we do next?",
    "New users are dropping off after the first session. Onboarding completion rate is only 30%. What retention action should we take?",
    "Power users who previously logged in daily have reduced usage to once a week over the past month. What do you recommend?",
    "We're seeing high churn among users who signed up during a promotional campaign. Many never used the core feature. What should we do?",
    "A cohort of users hasn't opened the app in 30 days but they have high lifetime value. Budget is available. What's the best action?",
    "Users are engaging with content but not converting to paid plans. Trial-to-paid rate is 5%. What retention approach works here?",
    "A segment of users complains frequently in support tickets but keeps using the product. Engagement is stable. What should we do?",
    "First-week retention is 60% but drops to 20% by week four. We don't know which features correlate with retention. What's the move?",
    "Push notification open rates have dropped 40% over 3 months for our most active cohort. What should we change?",
    "Users in a specific geography are churning at 2x the global rate. We have localized content but limited local support. Recommendations?",
    "A B2B SaaS cohort has low feature adoption across 3 key modules. Account managers report confusion during onboarding. What action?",
    "Seasonal users return every December but churn by February. We want to extend their lifecycle. What's the strategy?",
    "Free tier users who hit usage limits either churn or upgrade. 70% churn. How should we intervene before they hit the wall?",
    "Our reactivation emails have a 2% conversion rate. A cohort of 50k lapsed users hasn't engaged in 60+ days. Worth re-engaging?",
    "We launched a new feature but adoption is only 8% after 2 weeks. Existing users seem unaware of it. What should we do?",
]
