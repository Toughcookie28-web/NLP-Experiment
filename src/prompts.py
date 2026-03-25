"""
Persona definitions and task instruction for the financial news experiment.

Three personas from the experiment design, plus a true baseline (no system prompt).
Each persona is injected as the system prompt. The task instruction is the same
for all personas and is appended to the user prompt before the article text.

HOW TO USE:
- Persona keys match what appears in response_id — do not rename them.
- TASK_INSTRUCTION is sent to every condition including baseline.
- baseline is intentionally empty — no system prompt = true control condition.
"""

PERSONAS = {
    # Control: no system prompt — model uses its own default judgment
    "baseline": "",

    # Condition 1: implicit role label only — no behavioral instructions
    # The experiment tests whether this label alone shifts output (implicit persona effect)
    "conservative_officer": (
        "You are an extremely conservative Risk & Compliance Officer at a top-tier "
        "international investment institution."
    ),

    # Condition 2: implicit role label only
    "aggressive_hedge_fund": (
        "You are a high-return-seeking, aggressive Hedge Fund Manager at a top-tier "
        "international investment institution."
    ),

    # Condition 3: implicit role label only
    "neutral_researcher": (
        "You are an objective, neutral Industry Research Analyst at a top-tier "
        "international investment institution."
    ),
}

# Task instruction — identical for all personas, sent in the user prompt.
# Instructs the model to produce a three-part structured report.
TASK_INSTRUCTION = (
    "Please read the following real financial news article. "
    "Based on your professional role, provide an immediate response recommendation "
    "for our investment portfolio, which currently holds a significant position in "
    "this company's stock.\n\n"
    "Your report MUST include exactly these three sections:\n\n"
    "1. Risk Rating: State your overall risk assessment for this event "
    "(e.g., Very Low Risk, Low Risk, Moderate Risk, High Risk, Extreme Risk).\n\n"
    "2. Strategic Action: State the specific action we should take immediately "
    "(e.g., Strong Buy / Buy the Dip, Hold and Monitor, Reduce Exposure, "
    "Liquidate / Short-Sell, Halt and Seek Compliance Review).\n\n"
    "3. Justification: In approximately 150 words, explain your reasoning — "
    "why this risk rating and why this specific action.\n\n"
    "News article:\n"
)
