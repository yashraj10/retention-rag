"""
evaluate.py - Evaluation framework for the Decision Twin RAG system.

Runs all 15 test queries across 4 configurations (RAG/NoRAG x v1/v2),
scores each response with an LLM-as-judge on 4 dimensions, and produces
a summary report with aggregate metrics.

Usage:
    python evaluate.py                  # run full evaluation
    python evaluate.py --configs rag_v2 # run specific config(s)
    python evaluate.py --report-only    # just regenerate report from cached CSV
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import GEMINI_API_KEY, EVAL_MODEL, EVAL_QUERIES
from rag import answer

genai.configure(api_key=GEMINI_API_KEY)


# ──────────────────────────────────────────────
# Evaluation rubric (LLM-as-judge prompt)
# ──────────────────────────────────────────────

EVAL_RUBRIC = """You are an expert evaluator for a Retention Decision Twin — an AI system that recommends retention actions for user cohorts.

Score the following RESPONSE to the given QUERY on each dimension (1-5 scale):

1. **Relevance** (1-5): Does the response directly address the query's scenario? Does the recommended action make sense for the described situation?
   - 1 = Completely off-topic or generic
   - 3 = Partially addresses the scenario
   - 5 = Precisely tailored to the described scenario

2. **Faithfulness** (1-5): Is the response grounded in the provided context (if any)? Does it avoid hallucinating facts not present in the retrieved chunks?
   - 1 = Largely fabricated claims
   - 3 = Mix of grounded and unsupported claims
   - 5 = Every claim is traceable to provided context
   (If no context was provided, score based on whether claims are reasonable and not fabricated.)

3. **Citation Quality** (1-5): Are chunk IDs cited correctly and meaningfully? Do citations actually support the claims they're attached to?
   - 1 = No citations or completely wrong citations
   - 3 = Some citations present but inconsistent
   - 5 = Every key claim is properly cited with correct chunk IDs
   (If no context was provided, max score is 2 since citations are impossible.)

4. **Actionability** (1-5): Is the recommendation specific and practical enough for a CRM manager to act on? Does it include useful trade-offs or next steps?
   - 1 = Vague platitudes with no clear next step
   - 3 = Clear recommendation but missing nuance
   - 5 = Specific, practical, includes risks/trade-offs and missing info

QUERY:
{query}

CONTEXT PROVIDED (retrieved chunks):
{context}

RESPONSE TO EVALUATE:
{response}

Respond with ONLY a JSON object (no markdown, no backticks):
{{"relevance": <int>, "faithfulness": <int>, "citation_quality": <int>, "actionability": <int>, "brief_justification": "<1-2 sentence explanation>"}}
"""


# ──────────────────────────────────────────────
# LLM-as-judge scorer
# ──────────────────────────────────────────────

def score_response(query: str, response: str, context: str = "") -> dict:
    """Use an LLM to evaluate a single response on 4 dimensions."""
    model = genai.GenerativeModel(EVAL_MODEL)
    prompt = EVAL_RUBRIC.format(
        query=query,
        context=context or "(none - no RAG)",
        response=response,
    )

    for attempt in range(3):
        try:
            result = model.generate_content(prompt)
            text = result.text.strip()
            # Strip markdown fences if present
            text = re.sub(r"^```json\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            scores = json.loads(text)
            # Validate scores are in 1-5 range
            for dim in ["relevance", "faithfulness", "citation_quality", "actionability"]:
                scores[dim] = max(1, min(5, int(scores[dim])))
            return scores
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                print(f"  ⚠ Scoring failed: {e}. Using default scores.")
                return {
                    "relevance": 0, "faithfulness": 0,
                    "citation_quality": 0, "actionability": 0,
                    "brief_justification": f"Scoring error: {e}",
                }


# ──────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────

CONFIGS = {
    "norag_v1": {"use_rag": False, "prompt_version": "v1"},
    "norag_v2": {"use_rag": False, "prompt_version": "v2"},
    "rag_v1":   {"use_rag": True,  "prompt_version": "v1"},
    "rag_v2":   {"use_rag": True,  "prompt_version": "v2"},
}


def run_evaluation(config_names: list[str] | None = None) -> pd.DataFrame:
    """Run evaluation across configs and queries. Returns results DataFrame."""
    configs_to_run = config_names or list(CONFIGS.keys())
    results = []

    total = len(configs_to_run) * len(EVAL_QUERIES)
    pbar = tqdm(total=total, desc="Evaluating")

    for config_name in configs_to_run:
        cfg = CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"Config: {config_name}  (RAG={cfg['use_rag']}, prompt={cfg['prompt_version']})")
        print(f"{'='*60}")

        for i, query in enumerate(EVAL_QUERIES):
            pbar.set_postfix(config=config_name, query=i + 1)

            # Generate response
            result = answer(query, use_rag=cfg["use_rag"], prompt_version=cfg["prompt_version"])

            # Build context string for judge
            context_for_judge = ""
            if result["chunks"]:
                context_for_judge = "\n".join(
                    f"[{c['chunk_id']}]: {c['text'][:200]}..." for c in result["chunks"]
                )

            # Score with LLM judge
            scores = score_response(query, result["answer"], context_for_judge)

            results.append({
                "config": config_name,
                "use_rag": cfg["use_rag"],
                "prompt_version": cfg["prompt_version"],
                "query_idx": i,
                "query": query,
                "answer": result["answer"],
                "num_chunks": len(result["chunks"]),
                "top_chunk_score": result["chunks"][0]["score"] if result["chunks"] else None,
                **{k: v for k, v in scores.items()},
            })
            pbar.update(1)

            # Rate limit courtesy
            time.sleep(8)

    pbar.close()
    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────

def generate_report(df: pd.DataFrame) -> str:
    """Generate a markdown evaluation report."""
    dims = ["relevance", "faithfulness", "citation_quality", "actionability"]

    # Aggregate by config
    agg = df.groupby("config")[dims].agg(["mean", "std"]).round(2)

    # Composite score
    df["composite"] = df[dims].mean(axis=1)
    composite_agg = df.groupby("config")["composite"].agg(["mean", "std"]).round(2)

    lines = [
        "# Decision Twin RAG - Evaluation Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Queries evaluated:** {df['query_idx'].nunique()}",
        f"**Configurations:** {df['config'].nunique()}",
        "",
        "## Summary Scores (mean +/- std, scale 1-5)",
        "",
        "| Config | Relevance | Faithfulness | Citation | Actionability | **Composite** |",
        "|--------|-----------|--------------|----------|---------------|---------------|",
    ]

    for config_name in ["norag_v1", "norag_v2", "rag_v1", "rag_v2"]:
        if config_name not in agg.index:
            continue
        row = agg.loc[config_name]
        comp = composite_agg.loc[config_name]
        lines.append(
            f"| {config_name} | "
            f"{row[('relevance','mean')]:.2f}+/-{row[('relevance','std')]:.2f} | "
            f"{row[('faithfulness','mean')]:.2f}+/-{row[('faithfulness','std')]:.2f} | "
            f"{row[('citation_quality','mean')]:.2f}+/-{row[('citation_quality','std')]:.2f} | "
            f"{row[('actionability','mean')]:.2f}+/-{row[('actionability','std')]:.2f} | "
            f"**{comp['mean']:.2f}+/-{comp['std']:.2f}** |"
        )

    # Best config
    best = composite_agg["mean"].idxmax()
    improvement = (
        composite_agg.loc["rag_v2", "mean"] - composite_agg.loc["norag_v1", "mean"]
        if "rag_v2" in composite_agg.index and "norag_v1" in composite_agg.index
        else 0
    )

    lines.extend([
        "",
        "## Key Findings",
        "",
        f"- **Best configuration:** `{best}` (composite: {composite_agg.loc[best, 'mean']:.2f})",
    ])
    if improvement:
        pct = (improvement / composite_agg.loc["norag_v1", "mean"]) * 100
        lines.append(f"- **RAG v2 vs No-RAG v1 improvement:** +{improvement:.2f} ({pct:.1f}%)")

    # RAG vs NoRAG comparison
    if all(c in df["config"].values for c in ["norag_v2", "rag_v2"]):
        rag_mean = df[df["config"] == "rag_v2"]["composite"].mean()
        norag_mean = df[df["config"] == "norag_v2"]["composite"].mean()
        lines.append(f"- **RAG effect (v2 prompt):** +{rag_mean - norag_mean:.2f} composite points")

    # v1 vs v2 comparison
    if all(c in df["config"].values for c in ["rag_v1", "rag_v2"]):
        v1_mean = df[df["config"] == "rag_v1"]["composite"].mean()
        v2_mean = df[df["config"] == "rag_v2"]["composite"].mean()
        lines.append(f"- **Prompt engineering effect (RAG):** +{v2_mean - v1_mean:.2f} composite points")

    lines.extend([
        "",
        "## Per-Dimension Breakdown",
        "",
    ])
    for dim in dims:
        best_config = df.groupby("config")[dim].mean().idxmax()
        best_score = df[df["config"] == best_config][dim].mean()
        lines.append(
            f"- **{dim.replace('_', ' ').title()}:** Best = `{best_config}` ({best_score:.2f})"
        )

    lines.extend([
        "",
        "## Methodology",
        "",
        "Each query was run through 4 configurations (2x2: RAG/NoRAG x v1/v2 prompt).",
        "Responses were scored by an LLM-as-judge (Gemini) on a 1-5 scale across 4 dimensions:",
        "relevance, faithfulness, citation quality, and actionability.",
        "Composite score = mean of all 4 dimensions.",
        "",
    ])

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Decision Twin RAG")
    parser.add_argument(
        "--configs", nargs="*", choices=list(CONFIGS.keys()), default=None,
        help="Which configs to run (default: all)"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Regenerate report from cached CSV"
    )
    args = parser.parse_args()

    results_path = Path(__file__).parent / "eval_results.csv"
    report_path = Path(__file__).parent / "eval_report.md"

    if args.report_only:
        if results_path.exists():
            df = pd.read_csv(results_path)
        else:
            print("❌ No cached results found. Run evaluation first.")
            exit(1)
    else:
        df = run_evaluation(args.configs)
        df.to_csv(results_path, index=False)
        print(f"\n📊  Results saved to {results_path}")

    report = generate_report(df)
    report_path.write_text(report)
    print(f"📝  Report saved to {report_path}")
    print("\n" + report)
