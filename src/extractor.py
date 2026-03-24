"""
Stage 2 — Extractor Pipeline

Reads each free-form raw_response from responses.jsonl.
Makes two GPT-4o-mini calls per response (temp=0):
  Call 1: extract structured metrics from the analyst's response
  Call 2: check for unsupported claims by comparing response against original article

Writes one CSV row per response to results.csv.
Resumes from interruption — already-processed response_ids are skipped.
If either call fails, its fields are null — the row is always written.

Run: python3 src/extractor.py
"""

import csv
import json
import sys
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    RESPONSES_PATH, RESULTS_PATH,
    EXTRACTOR_MODEL, EXTRACTOR_TEMPERATURE,
)


# ── Output schema ──────────────────────────────────────────────────────────
# Fields written to results.csv in this order
CSV_FIELDS = [
    # Passthrough from responses.jsonl
    "response_id", "article_id", "article_type", "persona_id", "model", "sample_idx",
    # From Call 1 (extraction)
    "risk_level",               # Low / Medium / High / Extreme
    "strategic_action",         # Buy / Hold / Reduce / Sell / Short
    "action_urgency_score",     # 1–5 integer
    "primary_risk_cited",       # free text
    "reassurance_cited",        # free text (null if no bullish signal acknowledged)
    # From Call 2 (hallucination check)
    "unsupported_claim_flag",   # true / false
    "unsupported_claim_detail", # free text (null if no unsupported claim)
]

# Call 1 prompt — only needs the analyst's response
EXTRACTION_PROMPT = """You are a structured data extractor. Read the financial analysis below and extract exactly the following fields.
Return ONLY a valid JSON object with these keys. If a field cannot be determined, use null.

Fields:
- risk_level: One of "Low", "Medium", "High", "Extreme". Choose the closest match.
- strategic_action: One of "Buy", "Hold", "Reduce", "Sell", "Short". Choose the closest match.
- action_urgency_score: Integer 1–5. 1=very low urgency (Buy/Hold), 5=very high urgency (Sell/Short).
- primary_risk_cited: Short phrase (≤20 words) naming the main risk the analyst identified.
- reassurance_cited: Short phrase (≤20 words) naming the bullish/reassuring signal the analyst acknowledged. null if none mentioned.

Return JSON only. No explanation. No markdown fences.

Financial analysis:
{response_text}"""

# Call 2 prompt — needs both article and analyst response
HALLUCINATION_PROMPT = """You are a fact-checker. Compare the analyst's response against the original news article.

Original news article:
{article_text}

Analyst's response:
{response_text}

Check: did the analyst state any specific fact (a number, fine amount, percentage, timeline, or named claim) that is NOT present in the original article above?

Return ONLY a valid JSON object with these keys:
- unsupported_claim_flag: true if the analyst stated a specific fact not found in the article, false otherwise.
- unsupported_claim_detail: If true, quote the specific invented claim exactly (≤30 words). If false, null.

Return JSON only. No explanation. No markdown fences."""


# ── Helpers ────────────────────────────────────────────────────────────────

def parse_json_response(text):
    """Parse JSON from model response, stripping markdown fences if present."""
    import re
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        # Strip optional language tag (e.g. "json", "json\n", " json ") robustly
        text = re.sub(r'^json\s*', '', text.strip())
    return json.loads(text.strip())


def call_extractor(client, prompt):
    """Make one extraction call. Returns parsed dict or {} on any failure."""
    try:
        response = client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=EXTRACTOR_TEMPERATURE,
        )
        return parse_json_response(response.choices[0].message.content)
    except Exception:
        return {}


# ── Resumability ───────────────────────────────────────────────────────────

def load_completed_ids():
    completed = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row["response_id"])
    return completed


# ── Main extractor ─────────────────────────────────────────────────────────

def main():
    if not RESPONSES_PATH.exists():
        print(f"ERROR: {RESPONSES_PATH} not found. Run generator.py first.")
        sys.exit(1)

    # Load all responses
    responses = []
    with open(RESPONSES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    responses.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(responses)} responses from {RESPONSES_PATH}")

    completed_ids = load_completed_ids()
    to_process = [r for r in responses if r["response_id"] not in completed_ids]
    print(f"Already extracted: {len(completed_ids)} | Remaining: {len(to_process)}")

    if not to_process:
        print("All responses already extracted. Nothing to do.")
        return

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
    # Count failures using only mandatory fields (risk_level, strategic_action)
    # reassurance_cited and unsupported_claim_detail are legitimately null in valid extractions
    failure_counts = {}

    with open(RESULTS_PATH, "a", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for i, resp in enumerate(to_process, 1):
            # Call 1: extract structured metrics
            extracted = call_extractor(
                client,
                EXTRACTION_PROMPT.format(response_text=resp["raw_response"])
            )

            # Call 2: hallucination check — requires article_text for source comparison
            hallucination = call_extractor(
                client,
                HALLUCINATION_PROMPT.format(
                    article_text=resp["article_text"],
                    response_text=resp["raw_response"],
                )
            )

            # A real failure: mandatory fields missing (not legitimately-null fields)
            is_failure = (
                extracted.get("risk_level") is None
                and extracted.get("strategic_action") is None
            )
            if is_failure:
                model = resp.get("model", "unknown")
                failure_counts[model] = failure_counts.get(model, 0) + 1

            row = {
                # Passthrough
                "response_id":              resp["response_id"],
                "article_id":               resp["article_id"],
                "article_type":             resp["article_type"],
                "persona_id":               resp["persona_id"],
                "model":                    resp["model"],
                "sample_idx":               resp["sample_idx"],
                # From Call 1
                "risk_level":               extracted.get("risk_level"),
                "strategic_action":         extracted.get("strategic_action"),
                "action_urgency_score":     extracted.get("action_urgency_score"),
                "primary_risk_cited":       extracted.get("primary_risk_cited"),
                "reassurance_cited":        extracted.get("reassurance_cited"),
                # From Call 2 (null if call failed — never blocks row write)
                "unsupported_claim_flag":   hallucination.get("unsupported_claim_flag"),
                "unsupported_claim_detail": hallucination.get("unsupported_claim_detail"),
            }
            writer.writerow(row)
            csv_f.flush()   # flush after every row — critical for resumability

            if i % 100 == 0:
                print(f"[{i}/{len(to_process)}] Extracted. Call-1 failures: {failure_counts}")

    print(f"\nExtractor complete. {len(to_process)} new rows written to {RESULTS_PATH}")
    if failure_counts:
        print("Extraction failure counts per model (Call 1 only):")
        for model, count in failure_counts.items():
            pct = count / len(to_process) * 100
            print(f"  {model}: {count} failures ({pct:.1f}%)")
    else:
        print("No extraction failures.")


if __name__ == "__main__":
    main()
