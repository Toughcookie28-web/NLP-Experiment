"""
Persona definitions and task instruction for the financial news experiment.

HOW TO FILL IN:
- Each PERSONAS value is the full system prompt for that persona.
- TASK_INSTRUCTION is appended to the user prompt before the article text.
- Do NOT modify the keys in PERSONAS.
- The generator will refuse to run if any value is an empty string.
"""

PERSONAS = {
    "baseline": "",                   # intentionally empty — no system prompt for control
    "regulatory_analyst": "",         # FILL IN: Regulatory Precedent Analyst persona
    "market_signal_reader": "",       # FILL IN: Market Signal Reader persona
    "fundamentals_analyst": "",       # FILL IN: Business Fundamentals Analyst persona
}

TASK_INSTRUCTION = ""                 # FILL IN: the task instruction shown to all personas
