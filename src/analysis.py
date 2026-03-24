"""
Stage 3 — Analysis Pipeline

Joins responses.jsonl and results.csv. Validates coverage.
Computes five modular metrics. Saves outputs to data/figures/.

Run: python3 src/analysis.py

METRIC ARCHITECTURE:
Each metric function is completely independent — takes the merged DataFrame,
returns a result DataFrame or dict. You can add or remove functions from the
METRICS list freely without breaking anything else.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import RESPONSES_PATH, RESULTS_PATH, FIGURES_DIR, COVERAGE_THRESHOLD


# ── Join Validation ────────────────────────────────────────────────────────

def load_and_validate():
    """
    Load responses.jsonl and results.csv, join them, and validate coverage.
    Aborts with a clear message if coverage is below COVERAGE_THRESHOLD.
    """
    # Load responses
    responses = []
    with open(RESPONSES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    responses.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    df_resp = pd.DataFrame(responses)
    N = len(df_resp)
    print(f"responses.jsonl: {N} rows")

    # Load results
    df_res = pd.read_csv(RESULTS_PATH)
    M = len(df_res)
    print(f"results.csv:     {M} rows")

    # Coverage check — based on matched IDs, not just row count
    # (results.csv could have equal rows but different IDs — that would silently lose data)
    matched_ids = set(df_resp["response_id"]) & set(df_res["response_id"])
    missing_ids = set(df_resp["response_id"]) - matched_ids
    coverage = len(matched_ids) / N if N > 0 else 1.0

    if missing_ids:
        print(f"\nWARNING: {len(missing_ids)} responses not yet extracted ({len(missing_ids)/N*100:.1f}% missing)")
        print("Missing counts per model:")
        missing_df = df_resp[df_resp["response_id"].isin(missing_ids)]
        print(missing_df["model"].value_counts().to_string())

    if coverage < COVERAGE_THRESHOLD:
        print(f"\nCoverage {coverage:.1%} < threshold {COVERAGE_THRESHOLD:.1%}.")
        print("Re-run extractor.py before running analysis.")
        sys.exit(1)
    elif missing_ids:
        print(f"Coverage {coverage:.1%} is above threshold. Proceeding with available data.")

    # Left join — keep all responses, even if extraction produced nulls
    merged = df_resp.merge(df_res, on="response_id", how="left",
                           suffixes=("", "_extracted"))
    assert len(merged) == N, f"Join dropped rows: {N} -> {len(merged)}"

    # Null report
    extracted_cols = [
        "risk_level", "strategic_action", "action_urgency_score",
        "primary_risk_cited", "reassurance_cited",
        "unsupported_claim_flag", "unsupported_claim_detail",
    ]
    print("\nNull counts per extracted field:")
    for col in extracted_cols:
        if col in merged.columns:
            n_null = merged[col].isna().sum()
            print(f"  {col}: {n_null} nulls ({n_null/N*100:.1f}%)")

    return merged


# ── Metric Functions ───────────────────────────────────────────────────────
# Each function: takes merged DataFrame, saves figure(s) to FIGURES_DIR,
# prints a summary. Independent — removing one does not affect any other.

def compute_flip_rate(df):
    """
    Flip rate: % of samples where risk_level differs from the mode (most common),
    grouped by article × persona × model.
    Higher = more unstable.
    """
    print("\n=== Flip Rate ===")
    df_clean = df.dropna(subset=["risk_level"])

    def flip_rate_group(group):
        mode = group["risk_level"].mode()
        if mode.empty:
            return np.nan
        modal_value = mode.iloc[0]
        return (group["risk_level"] != modal_value).mean()

    result = (
        df_clean
        .groupby(["article_id", "persona_id", "model"])
        .apply(flip_rate_group, include_groups=False)
        .reset_index(name="flip_rate")
    )

    summary = result.groupby(["persona_id", "model"])["flip_rate"].mean().unstack("model")
    print(summary.to_string())

    # Plot: mean flip rate by persona, heatmap
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(summary, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Mean Flip Rate by Persona and Model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "flip_rate_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved: flip_rate_heatmap.png")
    return result


def compute_entropy(df):
    """
    Decision entropy: H = -Σ p(d) log₂ p(d) over risk_level distribution
    across 20 samples, grouped by article × persona × model.
    Higher = more spread/uncertain.
    """
    print("\n=== Decision Entropy ===")
    df_clean = df.dropna(subset=["risk_level"])
    RISK_LEVELS = ["Low", "Medium", "High", "Extreme"]

    def group_entropy(group):
        counts = group["risk_level"].value_counts()
        probs = np.array([counts.get(lv, 0) for lv in RISK_LEVELS], dtype=float)
        if probs.sum() == 0:
            return np.nan
        probs /= probs.sum()
        return scipy_entropy(probs, base=2)

    result = (
        df_clean
        .groupby(["article_id", "persona_id", "model"])
        .apply(group_entropy, include_groups=False)
        .reset_index(name="entropy")
    )

    summary = result.groupby(["persona_id", "model"])["entropy"].mean().unstack("model")
    print(summary.to_string())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(summary, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Mean Decision Entropy by Persona and Model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "entropy_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved: entropy_heatmap.png")
    return result


def compute_conservatism_score(df):
    """
    Conservatism score: mean action_urgency_score per persona × model.
    Higher urgency score = more conservative (sell/reduce) response.
    """
    print("\n=== Conservatism Score ===")
    df_clean = df.dropna(subset=["action_urgency_score"])
    df_clean = df_clean.copy()
    df_clean["action_urgency_score"] = pd.to_numeric(
        df_clean["action_urgency_score"], errors="coerce"
    )

    result = (
        df_clean
        .groupby(["persona_id", "model"])["action_urgency_score"]
        .mean()
        .unstack("model")
    )
    print(result.to_string())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(result, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax)
    ax.set_title("Mean Conservatism Score (Action Urgency) by Persona and Model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "conservatism_score_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved: conservatism_score_heatmap.png")
    return result


def compute_semantic_variance(df):
    """
    Semantic variance: mean pairwise cosine distance of raw_response embeddings
    across 20 samples, grouped by article × persona × model.
    Higher = more divergent reasoning even when decisions are the same.

    Uses sentence-transformers for embeddings + pure numpy for cosine distance.
    No scikit-learn dependency.
    """
    print("\n=== Semantic Variance ===")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Skipping semantic variance.")
        print("Install with: pip install sentence-transformers")
        return None

    model = SentenceTransformer("all-MiniLM-L6-v2")   # fast, small, adequate for this task

    def mean_pairwise_cosine_dist(texts):
        if len(texts) < 2:
            return np.nan
        embeddings = model.encode(texts, show_progress_bar=False)
        # Pure numpy cosine distance — no sklearn needed
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.where(norms == 0, 1, norms)   # avoid div-by-zero
        similarity_matrix = normalized @ normalized.T
        dist_matrix = 1.0 - similarity_matrix
        # Upper triangle only (avoid double-counting and self-distance)
        n = len(texts)
        upper = dist_matrix[np.triu_indices(n, k=1)]
        return float(np.mean(upper))

    results = []
    groups = df.dropna(subset=["raw_response"]).groupby(
        ["article_id", "persona_id", "model"]
    )
    total = len(groups)
    for i, (key, group) in enumerate(groups):
        if i % 50 == 0:
            print(f"  Embedding group {i}/{total}...")
        texts = group["raw_response"].tolist()
        dist = mean_pairwise_cosine_dist(texts)
        results.append({
            "article_id": key[0],
            "persona_id": key[1],
            "model": key[2],
            "semantic_variance": dist,
        })

    result = pd.DataFrame(results)
    summary = result.groupby(["persona_id", "model"])["semantic_variance"].mean().unstack("model")
    print(summary.to_string())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(summary, annot=True, fmt=".3f", cmap="Purples", ax=ax)
    ax.set_title("Mean Semantic Variance by Persona and Model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "semantic_variance_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved: semantic_variance_heatmap.png")
    return result


def compute_unsupported_claim_rate(df):
    """
    Unsupported claim rate: % of responses where unsupported_claim_flag = True,
    grouped by persona × model.
    Higher = this persona makes the model more likely to invent facts.
    """
    print("\n=== Unsupported Claim Rate ===")
    df_clean = df.dropna(subset=["unsupported_claim_flag"]).copy()
    # Normalize to boolean — extractor may return string "true"/"false"
    df_clean["hallucination"] = df_clean["unsupported_claim_flag"].apply(
        lambda x: str(x).lower() in ("true", "1", "yes")
    )

    result = (
        df_clean
        .groupby(["persona_id", "model"])["hallucination"]
        .mean()
        .unstack("model")
    )
    print(result.to_string())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(result, annot=True, fmt=".2%", cmap="Reds", ax=ax)
    ax.set_title("Unsupported Claim Rate by Persona and Model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "unsupported_claim_rate_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved: unsupported_claim_rate_heatmap.png")
    return result


# ── Metric Registry ────────────────────────────────────────────────────────
# Add or remove functions here freely — nothing else depends on this list

METRICS = [
    compute_flip_rate,
    compute_entropy,
    compute_conservatism_score,
    compute_semantic_variance,
    compute_unsupported_claim_rate,
]


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not RESPONSES_PATH.exists():
        print(f"ERROR: {RESPONSES_PATH} not found. Run generator.py first.")
        sys.exit(1)
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run extractor.py first.")
        sys.exit(1)

    merged = load_and_validate()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for metric_fn in METRICS:
        try:
            metric_fn(merged)
        except Exception as e:
            print(f"WARNING: {metric_fn.__name__} failed: {e}")
            # Continue with other metrics — modular, independent

    print(f"\nAnalysis complete. Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
