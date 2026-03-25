"""
Stage 1 — Generator Pipeline

Sends each article × persona × model combination to OpenRouter SAMPLES_PER_CONDITION times.
Writes one JSON line to responses.jsonl immediately after each API call (never batched).
Resumes safely from any interruption — already-completed response_ids are skipped.

Run: python3 src/generator.py
"""

import json
import sys
import time
from pathlib import Path
from openai import OpenAI
import tiktoken

# Add project root to path so we can import config and prompts
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    ARTICLES_PATH, RESPONSES_PATH,
    MODELS, MODEL_PARAMS, SAMPLES_PER_CONDITION,
    GPT4O_ID, O3MINI_ID, SMALL_MODEL_ID,
)
from src.prompts import PERSONAS, TASK_INSTRUCTION

RETRY_SLEEP_SECONDS = 2   # wait between retried calls after an API error

# ── Guard: validate config before any API calls ────────────────────────────

def validate_config():
    if SMALL_MODEL_ID == "FILL_IN_BEFORE_RUNNING":
        print("ERROR: SMALL_MODEL_ID not set in src/config.py")
        print("Fill in a real OpenRouter model ID (e.g. 'meta-llama/llama-3.1-8b-instruct').")
        sys.exit(1)


# ── Guard: refuse to run with empty prompts ────────────────────────────────

def validate_prompts():
    errors = []
    for persona_id, persona_text in PERSONAS.items():
        if persona_id == "baseline":
            continue                        # baseline is intentionally empty
        if not persona_text.strip():
            errors.append(f"  PERSONAS['{persona_id}'] is empty")
    if not TASK_INSTRUCTION.strip():
        errors.append("  TASK_INSTRUCTION is empty")
    if errors:
        print("ERROR: Cannot run generator — the following prompts are not filled in:")
        for e in errors:
            print(e)
        print("\nEdit src/prompts.py and fill in all persona descriptions and TASK_INSTRUCTION.")
        sys.exit(1)


# ── Guard: validate articles file ─────────────────────────────────────────

def load_articles():
    if not ARTICLES_PATH.exists():
        print(f"ERROR: Articles file not found at {ARTICLES_PATH}")
        print("Place articles.jsonl in data/news/ before running.")
        sys.exit(1)

    required_fields = {"article_id", "article_text"}

    try:
        with open(ARTICLES_PATH) as f:
            articles = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in articles.json: {e}")
        sys.exit(1)

    for i, record in enumerate(articles, 1):
        missing = required_fields - set(record.keys())
        if missing:
            print(f"ERROR: Article {i} missing fields: {missing}")
            sys.exit(1)

    if not articles:
        print("ERROR: articles.json is empty.")
        sys.exit(1)

    print(f"Loaded {len(articles)} articles from {ARTICLES_PATH}")
    return articles


# ── Resumability: load already-completed response IDs ─────────────────────

def load_completed_ids():
    completed = set()
    if RESPONSES_PATH.exists():
        with open(RESPONSES_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    completed.add(record["response_id"])
                except (json.JSONDecodeError, KeyError):
                    pass    # skip malformed lines silently
    return completed


# ── Response ID ───────────────────────────────────────────────────────────

def make_response_id(article_id, persona_id, model_id, sample_idx):
    model_slug = model_id.replace("/", "-")
    return f"{article_id}__{persona_id}__{model_slug}__{sample_idx:02d}"


# ── Token counting (for cost estimation) ──────────────────────────────────

def count_tokens(text, model="gpt-4o"):
    """Approximate token count — uses gpt-4o encoding for all models."""
    # Uses gpt-4o encoding for all models — close enough for rough cost estimates
    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ── Cost estimation (rough) ────────────────────────────────────────────────
# Prices in USD per 1M tokens (input / output) — update if OpenRouter prices change
COST_PER_1M = {
    GPT4O_ID:  {"input": 2.50, "output": 10.00},
    O3MINI_ID: {"input": 1.10, "output": 4.40},
    "default": {"input": 0.20, "output": 0.80},    # fallback for small model
}

def estimate_cost(model_id, prompt_tokens, completion_tokens):
    rates = COST_PER_1M.get(model_id, COST_PER_1M["default"])
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000


# ── Main generator ─────────────────────────────────────────────────────────

def main():
    validate_config()
    validate_prompts()
    articles = load_articles()
    completed_ids = load_completed_ids()

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    # Build the full work list
    work_items = []
    for article in articles:
        for persona_id in PERSONAS:
            for model_id in MODELS:
                for sample_idx in range(SAMPLES_PER_CONDITION):
                    response_id = make_response_id(
                        article["article_id"], persona_id, model_id, sample_idx
                    )
                    if response_id not in completed_ids:
                        work_items.append((article, persona_id, model_id, sample_idx, response_id))

    total = len(articles) * len(PERSONAS) * len(MODELS) * SAMPLES_PER_CONDITION
    already_done = total - len(work_items)
    print(f"Total calls: {total} | Already done: {already_done} | Remaining: {len(work_items)}")

    if not work_items:
        print("All calls already complete. Nothing to do.")
        return

    # Ensure output directory exists
    RESPONSES_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Running cost totals per model
    cost_by_model = {m: 0.0 for m in MODELS}
    calls_done = 0

    with open(RESPONSES_PATH, "a") as out_f:
        for article, persona_id, model_id, sample_idx, response_id in work_items:
            # Build messages
            system_prompt = PERSONAS[persona_id]    # empty string for baseline = no system message
            # Prepend headline if present — gives the model the same framing a real analyst has
            headline = article.get("headline", "")
            article_block = (
                f"Headline: {headline}\n\n{article['article_text']}"
                if headline else article["article_text"]
            )
            user_prompt = f"{TASK_INSTRUCTION}\n\n{article_block}"

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            # Count input tokens for cost estimation
            full_prompt_text = (system_prompt + "\n" if system_prompt else "") + user_prompt
            prompt_tokens_est = count_tokens(full_prompt_text)

            # Build call kwargs — model-specific params (e.g. o3-mini has no temperature)
            call_kwargs = {
                "model": model_id,
                "messages": messages,
                **MODEL_PARAMS[model_id],
            }

            try:
                response = client.chat.completions.create(**call_kwargs)
                raw_response = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens if response.usage else prompt_tokens_est
                completion_tokens = response.usage.completion_tokens if response.usage else 0
            except Exception as e:
                print(f"  ERROR on {response_id}: {e}")
                # Log the error inline but continue — don't crash the whole run
                # This response_id will be missing and retried on next run
                time.sleep(RETRY_SLEEP_SECONDS)
                continue

            cost = estimate_cost(model_id, prompt_tokens, completion_tokens)
            cost_by_model[model_id] += cost

            # Write immediately — never batch
            record = {
                "response_id":        response_id,
                "article_id":         article["article_id"],
                "persona_id":         persona_id,
                "model":              model_id,
                "sample_idx":         sample_idx,
                "raw_response":       raw_response,
                "article_text":       article["article_text"],
                "prompt_tokens":      prompt_tokens,
                "completion_tokens":  completion_tokens,
                "estimated_cost_usd": round(cost, 6),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()   # flush after every write — critical for resumability

            calls_done += 1

            # Progress report every 100 calls
            if calls_done % 100 == 0:
                total_cost = sum(cost_by_model.values())
                already_done_total = already_done + calls_done
                print(
                    f"[Call {already_done_total}/{total}] "
                    f"Running cost: ${total_cost:.2f} | "
                    + " | ".join(
                        f"{m.split('/')[-1]}: ${c:.2f}"
                        for m, c in cost_by_model.items()
                        if c > 0
                    )
                )

    total_cost = sum(cost_by_model.values())
    print(f"\nGenerator complete. {calls_done} new calls. Session cost: ${total_cost:.2f}")


if __name__ == "__main__":
    main()
