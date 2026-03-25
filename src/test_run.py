"""
Smoke test — run 3 real API calls before launching the full experiment.

Tests ONE article × 3 personas (baseline, conservative_officer, aggressive_hedge_fund)
× GPT-4o × 1 sample. Then immediately runs the extractor on those 3 responses
so you can verify the full pipeline end-to-end.

Total cost: ~$0.05. Saves outputs to data/test/ (separate from main data dirs).

Run: python3 src/test_run.py
"""

import json
import sys
import csv
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    ARTICLES_PATH, MODEL_PARAMS, GPT4O_ID,
    EXTRACTOR_MODEL, EXTRACTOR_TEMPERATURE,
)
from src.prompts import PERSONAS, TASK_INSTRUCTION
from src.generator import validate_config, validate_prompts, load_articles
from src.extractor import (
    EXTRACTOR_SYSTEM_PROMPT, EXTRACTION_PROMPT, HALLUCINATION_PROMPT,
    call_extractor, CSV_FIELDS,
)

# ── Test config ────────────────────────────────────────────────────────────
TEST_DIR       = Path(__file__).parent.parent / "data" / "test"
TEST_RESPONSES = TEST_DIR / "test_responses.jsonl"
TEST_RESULTS   = TEST_DIR / "test_results.csv"

# Which article to test on (index into articles list)
TEST_ARTICLE_IDX = 0

# Which personas to test — pick the three that show the most contrast
TEST_PERSONAS = ["baseline", "conservative_officer", "aggressive_hedge_fund"]

# Only test with GPT-4o (fastest feedback, most reliable output format)
TEST_MODEL = GPT4O_ID


def run_generator_test(client, articles):
    article = articles[TEST_ARTICLE_IDX]
    headline = article.get("headline", "")
    article_block = (
        f"Headline: {headline}\n\n{article['article_text']}"
        if headline else article["article_text"]
    )

    print(f"\n{'='*60}")
    print(f"TEST ARTICLE: {article['article_id']}")
    print(f"Headline: {headline}")
    print(f"{'='*60}\n")

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    responses = []

    with open(TEST_RESPONSES, "w") as f:
        for persona_id in TEST_PERSONAS:
            system_prompt = PERSONAS[persona_id]
            user_prompt = f"{TASK_INSTRUCTION}\n\n{article_block}"

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            print(f"--- Calling GPT-4o | persona: {persona_id} ---")
            try:
                resp = client.chat.completions.create(
                    model=TEST_MODEL,
                    messages=messages,
                    **MODEL_PARAMS[TEST_MODEL],
                )
                raw_response = resp.choices[0].message.content
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            response_id = f"{article['article_id']}__{persona_id}__gpt-4o__00"
            record = {
                "response_id":  response_id,
                "article_id":   article["article_id"],
                "persona_id":   persona_id,
                "model":        TEST_MODEL,
                "sample_idx":   0,
                "raw_response": raw_response,
                "article_text": article["article_text"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            responses.append(record)

            # Print the full response so you can visually inspect it
            print(f"\n[RESPONSE — {persona_id}]")
            print(raw_response)
            print()

    print(f"\nGenerator test done. {len(responses)} responses saved to {TEST_RESPONSES}")
    return responses


def run_extractor_test(client, responses):
    print(f"\n{'='*60}")
    print("RUNNING EXTRACTOR ON TEST RESPONSES")
    print(f"{'='*60}\n")

    rows = []
    for resp in responses:
        raw_response = resp["raw_response"]

        extracted = call_extractor(
            client,
            EXTRACTOR_SYSTEM_PROMPT,
            EXTRACTION_PROMPT.format(response_text=raw_response),
        )
        hallucination = call_extractor(
            client,
            EXTRACTOR_SYSTEM_PROMPT,
            HALLUCINATION_PROMPT.format(
                article_text=resp["article_text"],
                response_text=raw_response,
            ),
        )

        row = {
            "response_id":   resp["response_id"],
            "article_id":    resp["article_id"],
            "persona_id":    resp["persona_id"],
            "model":         resp["model"],
            "sample_idx":    resp["sample_idx"],
            "risk_rating_score":              extracted.get("risk_rating_score"),
            "strategic_action":               extracted.get("strategic_action"),
            "action_urgency":                 extracted.get("action_urgency"),
            "compliance_refusal_flag":        extracted.get("compliance_refusal_flag"),
            "analysis_primary_focus":         extracted.get("analysis_primary_focus"),
            "reasoning_basis":                extracted.get("reasoning_basis"),
            "tone_confidence_level":          extracted.get("tone_confidence_level"),
            "risk_thesis_hook":               extracted.get("risk_thesis_hook"),
            "unsupported_financial_claim_flag": hallucination.get("unsupported_financial_claim_flag"),
            "output_word_count":              len(raw_response.split()),
        }
        rows.append(row)

        # Print a concise summary so you can see what was extracted
        print(f"[EXTRACTED — {resp['persona_id']}]")
        for key, val in row.items():
            if key not in ("response_id", "article_id", "model", "sample_idx"):
                print(f"  {key}: {val}")
        print()

    # Save to CSV
    with open(TEST_RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extractor test done. Results saved to {TEST_RESULTS}")


def main():
    validate_config()
    validate_prompts()
    articles = load_articles()

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    responses = run_generator_test(client, articles)
    if not responses:
        print("No responses generated — check API key and model access.")
        sys.exit(1)

    run_extractor_test(client, responses)

    print("\n" + "="*60)
    print("SMOKE TEST COMPLETE")
    print("Review the responses and extracted fields above.")
    print("If everything looks correct, run the full pipeline:")
    print("  python3 src/generator.py")
    print("  python3 src/extractor.py")
    print("  python3 src/analysis.py")
    print("="*60)


if __name__ == "__main__":
    main()
