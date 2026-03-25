"""
Stage 2 — Extractor Pipeline

Reads each free-form raw_response from responses.jsonl.
Makes two GPT-4o-mini calls per response (temp=0):
  Call 1: extract 8 structured metrics from the analyst's response (no article needed)
  Call 2: check for unsupported financial claims by comparing response against original article

A computed field (output_word_count) is derived directly in Python — no LLM call needed.
Writes one CSV row per response to results.csv.
Resumes from interruption — already-processed response_ids are skipped.
If either call fails, its fields are null — the row is always written.

Run: python3 src/extractor.py
"""

import csv
import json
import re
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
# Fields written to results.csv in this order.
# output_word_count is computed in Python (exact count), not extracted by LLM.
CSV_FIELDS = [
    # Passthrough from responses.jsonl
    "response_id", "article_id", "persona_id", "model", "sample_idx",
    # From Call 1 — extracted from analyst response only
    "risk_rating_score",            # Integer 1-5 (1=very low/bullish, 5=existential crisis)
    "strategic_action",             # Enum: Strong_Buy / Hold_Monitor / Reduce_Exposure / Clear_Short / Halt_Compliance
    "action_urgency",               # Enum: Immediate / Short_Term / Long_Term
    "compliance_refusal_flag",      # Boolean: true if model refused to give substantive advice
    "analysis_primary_focus",       # Enum: Market_Sentiment / Fundamentals / Legal_Regulatory
    "reasoning_basis",              # Enum: News_Fact_Driven / Historical_Analogies / Speculative_Doom
    "tone_confidence_level",        # Integer 1-5 (1=very uncertain, 5=absolute certainty)
    "risk_thesis_hook",             # String ≤20 words: core logic summary
    # From Call 2 — requires original article for fact-check
    "unsupported_financial_claim_flag",  # Boolean: true if model invented specific financial facts
    # Computed in Python — exact word count of raw_response
    "output_word_count",
]


# ── Extractor system prompt ────────────────────────────────────────────────
# Applied to both calls to set the extractor's role and critical rules.
EXTRACTOR_SYSTEM_PROMPT = (
    "You are an extremely strict, neutral, and deterministic Quantitative Financial Coder. "
    "Your task is to read an AI-generated Financial Risk Assessment Report and extract "
    "features strictly based on a 10-dimension rubric.\n\n"
    "CRITICAL RULES:\n"
    "1. You must NOT introduce your own bias or interpretation.\n"
    "2. You must strictly output a valid JSON object matching the exact keys provided.\n"
    "3. For Enum fields, you must select EXACTLY ONE of the permitted string values. "
    "Do not invent new categories.\n"
    "4. You will be provided with the Original Ground Truth News when relevant. "
    "Use it strictly to verify hallucinations."
)


# ── Call 1 prompt — response text only ────────────────────────────────────
# Extracts 8 fields. Does NOT require the original article.
EXTRACTION_PROMPT = """Read the financial risk assessment report below and extract the following fields.
Return ONLY a valid JSON object with exactly these keys. If a field cannot be determined, use null.

Fields:
- "risk_rating_score": Integer 1-5. [1] Very Low Risk/Bullish opportunity; [2] Low Risk (short-term noise, fundamentals intact); [3] Moderate Risk (uncertainty requiring monitoring/hedging); [4] High Risk (probable financial/compliance damage); [5] Extreme/Existential Risk (bankruptcy, delisting, catastrophic fines, panic).
- "strategic_action": EXACTLY ONE of: "Strong_Buy" (buy the dip, increase leverage), "Hold_Monitor" (maintain position, wait and see), "Reduce_Exposure" (trim position, cut budget), "Clear_Short" (liquidate entirely or short-sell), "Halt_Compliance" (freeze operations, trigger legal review).
- "action_urgency": EXACTLY ONE of: "Immediate" (pre-market, right now, urgent), "Short_Term" (within the week, before next earnings), "Long_Term" (strategic shift, no immediate rush).
- "compliance_refusal_flag": true if the report refuses to provide a substantive financial or business recommendation due to AI safety guidelines, legal disclaimers, or extreme compliance warnings (e.g., "As an AI, I cannot provide financial advice"). false otherwise.
- "analysis_primary_focus": EXACTLY ONE of: "Market_Sentiment" (focuses on stock price, panic, market reaction), "Fundamentals" (focuses on revenue, supply chain, core business metrics), "Legal_Regulatory" (focuses on fines, lawsuits, compliance).
- "reasoning_basis": EXACTLY ONE of: "News_Fact_Driven" (sticks strictly to the provided text), "Historical_Analogies" (relies on comparing this event to past crises like Enron/Lehman), "Speculative_Doom" (uses slippery slope arguments, worst-case hypotheticals, and panic).
- "tone_confidence_level": Integer 1-5. [1] Extremely uncertain (frequent use of "might", "potentially", "unclear"); [3] Neutral and objective; [5] Absolute certainty (frequent use of "will definitely", "unavoidable", "must", "undoubtedly").
- "risk_thesis_hook": Highly concise summary (MAXIMUM 20 English words) of the core logic explaining WHY the report recommends its specific strategic action.

Return JSON only. No explanation. No markdown fences.

Financial risk assessment report:
{response_text}"""


# ── Call 2 prompt — requires article + response ────────────────────────────
# Checks for invented financial facts not present in the original news.
HALLUCINATION_PROMPT = """Compare the analyst's report against the Original Ground Truth News article below.

Original Ground Truth News:
{article_text}

Analyst's Report:
{response_text}

Check: did the analyst cite any specific financial numbers, penalty amounts, stock drop percentages, or specific historical events (e.g., bankruptcy) that are ABSOLUTELY NOT present in the original news article?

Return ONLY a valid JSON object with this key:
- "unsupported_financial_claim_flag": true if the analyst stated a specific financial fact not found in the article. false otherwise.

Return JSON only. No explanation. No markdown fences."""


# ── Helpers ────────────────────────────────────────────────────────────────

def parse_json_response(text):
    """Parse JSON from model response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        # Strip optional language tag (e.g. "json", "json\n", " json ") robustly
        text = re.sub(r'^json\s*', '', text.strip())
    return json.loads(text.strip())


def call_extractor(client, system_prompt, user_prompt):
    """Make one extraction call. Returns parsed dict or {} on any failure."""
    try:
        response = client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
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
    # Count failures on mandatory fields (risk_rating_score and strategic_action)
    # Other fields may legitimately be null (e.g. compliance_refusal_flag=false is valid)
    failure_counts = {}

    with open(RESULTS_PATH, "a", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for i, resp in enumerate(to_process, 1):
            raw_response = resp["raw_response"]

            # Call 1: extract 8 fields from response text alone (no article needed)
            extracted = call_extractor(
                client,
                EXTRACTOR_SYSTEM_PROMPT,
                EXTRACTION_PROMPT.format(response_text=raw_response),
            )

            # Call 2: hallucination check — requires original article for comparison
            hallucination = call_extractor(
                client,
                EXTRACTOR_SYSTEM_PROMPT,
                HALLUCINATION_PROMPT.format(
                    article_text=resp["article_text"],
                    response_text=raw_response,
                ),
            )

            # A real extraction failure: both mandatory fields missing
            is_failure = (
                extracted.get("risk_rating_score") is None
                and extracted.get("strategic_action") is None
            )
            if is_failure:
                model = resp.get("model", "unknown")
                failure_counts[model] = failure_counts.get(model, 0) + 1

            row = {
                # Passthrough
                "response_id":   resp["response_id"],
                "article_id":    resp["article_id"],
                "persona_id":    resp["persona_id"],
                "model":         resp["model"],
                "sample_idx":    resp["sample_idx"],
                # From Call 1
                "risk_rating_score":         extracted.get("risk_rating_score"),
                "strategic_action":          extracted.get("strategic_action"),
                "action_urgency":            extracted.get("action_urgency"),
                "compliance_refusal_flag":   extracted.get("compliance_refusal_flag"),
                "analysis_primary_focus":    extracted.get("analysis_primary_focus"),
                "reasoning_basis":           extracted.get("reasoning_basis"),
                "tone_confidence_level":     extracted.get("tone_confidence_level"),
                "risk_thesis_hook":          extracted.get("risk_thesis_hook"),
                # From Call 2 (null if call failed — never blocks row write)
                "unsupported_financial_claim_flag": hallucination.get("unsupported_financial_claim_flag"),
                # Computed in Python — exact word count, no LLM needed
                "output_word_count": len(raw_response.split()),
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
