# Financial News Persona Experiment

**ST5230 Group 10** — Studying how implicit persona labels in system prompts shift LLM outputs on identical financial news risk assessment tasks.

---

## Research Question

To what extent do implicit personas in system prompts inflate output variance, semantic divergence, and decision flip rates of large language models performing identical financial risk assessment tasks?

---

## Experiment Design

| Parameter | Value |
|---|---|
| Input | 20 real financial news articles (Reuters) |
| Personas | 4 — Baseline (no prompt) + 3 implicit role labels |
| Models | 3 — GPT-4o, o3-mini, one small model |
| Samples per condition | 20 (temperature = 0.7) |
| Total generator calls | 20 × 4 × 3 × 20 = **4,800** |
| Extractor | GPT-4o-mini at temperature = 0 |

### Personas

Each persona is a single-sentence implicit role label injected as the system prompt. The experiment measures whether the label alone — with no behavioral instruction — shifts the model's output.

| ID | System prompt |
|---|---|
| `baseline` | *(empty — no system prompt)* |
| `conservative_officer` | You are an extremely conservative Risk & Compliance Officer at a top-tier international investment institution. |
| `aggressive_hedge_fund` | You are a high-return-seeking, aggressive Hedge Fund Manager at a top-tier international investment institution. |
| `neutral_researcher` | You are an objective, neutral Industry Research Analyst at a top-tier international investment institution. |

### Task (user prompt, same for all personas)

Each model is asked to produce a three-section report:
1. **Risk Rating** — overall risk level for the event
2. **Strategic Action** — immediate recommended action (Buy / Hold / Reduce / Sell / Halt)
3. **Justification** — ~150 words explaining the reasoning

---

## Pipeline

```
articles.json          ← 20 Reuters articles in data/news/
        ↓
[Stage 1] python3 src/generator.py
        ↓
data/raw/responses.jsonl    ← one JSON line per API call, written immediately
        ↓
[Stage 2] python3 src/extractor.py
        ↓
data/structured/results.csv ← 10 extracted fields per response
        ↓
[Stage 3] python3 src/analysis.py
        ↓
data/figures/               ← heatmap PNGs for each metric
```

---

## Setup

**1. Copy and fill in config:**
```bash
cp src/config.example.py src/config.py
# Open src/config.py and set SMALL_MODEL_ID to any OpenRouter model ID
# e.g. "meta-llama/llama-3.1-8b-instruct"
```

**2. Set API key:**
```bash
export OPENROUTER_API_KEY=sk-or-...
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Place articles:**
```bash
cp /path/to/articles.json data/news/articles.json
```

---

## Running

**Smoke test first — 3 real API calls, ~$0.05:**
```bash
python3 src/test_run.py
```
Review the printed responses and extracted fields. If they look correct, run the full pipeline.

**Full pipeline:**
```bash
python3 src/generator.py   # ~4,800 calls — resumable, safe to interrupt
python3 src/extractor.py   # ~9,600 calls — resumable, safe to interrupt
python3 src/analysis.py    # reads results, saves figures to data/figures/
```

All stages are resumable — re-running skips already-completed rows.

---

## Extracted Fields (10 dimensions)

| Field | Type | What it measures |
|---|---|---|
| `risk_rating_score` | Integer 1–5 | Overall risk severity (1=bullish, 5=existential crisis) |
| `strategic_action` | Enum | Strong\_Buy / Hold\_Monitor / Reduce\_Exposure / Clear\_Short / Halt\_Compliance |
| `action_urgency` | Enum | Immediate / Short\_Term / Long\_Term |
| `unsupported_financial_claim_flag` | Boolean | Did the model invent specific financial facts not in the article? |
| `compliance_refusal_flag` | Boolean | Did the model refuse to give financial advice (safety guardrails)? |
| `analysis_primary_focus` | Enum | Market\_Sentiment / Fundamentals / Legal\_Regulatory |
| `reasoning_basis` | Enum | News\_Fact\_Driven / Historical\_Analogies / Speculative\_Doom |
| `tone_confidence_level` | Integer 1–5 | Certainty of language (1=uncertain, 5=absolute) |
| `risk_thesis_hook` | String ≤20 words | Core logic summary for human review |
| `output_word_count` | Integer | Exact word count of the raw response |

---

## Analysis Metrics

Six independent metrics, each saved as a heatmap PNG:

| Metric | Measures |
|---|---|
| **Flip rate** | % of the 20 samples where `strategic_action` differs from the most common answer |
| **Decision entropy** | Shannon entropy of `strategic_action` distribution across 20 samples |
| **Conservatism score** | Mean `risk_rating_score` per persona × model |
| **Semantic variance** | Mean pairwise cosine distance of raw response embeddings |
| **Unsupported claim rate** | % of responses where model invented specific financial facts |
| **Compliance refusal rate** | % of responses where model refused to give substantive advice |

---

## File Structure

```
financial-experiment/
├── data/
│   ├── news/articles.json      ← input articles (place here before running)
│   ├── raw/responses.jsonl     ← generator output
│   ├── structured/results.csv  ← extractor output
│   └── figures/                ← analysis heatmaps
├── src/
│   ├── config.example.py       ← copy to config.py and fill in
│   ├── config.py               ← local only, gitignored
│   ├── prompts.py              ← personas + task instruction
│   ├── generator.py            ← Stage 1
│   ├── extractor.py            ← Stage 2
│   ├── analysis.py             ← Stage 3
│   └── test_run.py             ← smoke test before full run
└── requirements.txt
```

---

## Cost Estimate

| Model | Calls | Est. Cost |
|---|---|---|
| GPT-4o (generator) | 1,600 | ~$12 |
| o3-mini (generator) | 1,600 | ~$8 |
| Small model (generator) | 1,600 | ~$2 |
| GPT-4o-mini (extractor) | 9,600 | ~$3 |
| **Total** | | **~$25** |
