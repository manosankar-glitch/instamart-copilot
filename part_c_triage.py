
from pathlib import Path
import json
import random
import re
import time

import pandas as pd
import mock_llm


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
TOKEN_BUDGET = 6000

ALLOWED = ["billing", "delivery", "technical", "fraud"]

# Apply these patterns in exactly this order.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")

PII_PATTERNS = [
    ("emails", EMAIL_RE, "[EMAIL]"),
    ("phones", PHONE_RE, "[PHONE]"),
    ("order_ids", ORDER_RE, "[ORDER_ID]"),
]

ZERO_SHOT_PROMPT = """
CLASSIFY the customer's support ticket into exactly one allowed category:
billing, delivery, technical, fraud.

Return JSON only, with exactly this form:
{"category": "...", "reason": "..."}

The category must be one of: billing, delivery, technical, fraud.
Use the ticket text as the evidence. Do not invent facts.
Do not include Markdown fences or any text outside the JSON object.
""".strip()

FEW_SHOT_PROMPT = """
CLASSIFY the customer's support ticket into exactly one allowed category:
billing, delivery, technical, fraud.

Return JSON only, with exactly this form:
{"category": "...", "reason": "..."}

The category must be one of: billing, delivery, technical, fraud.
Use the ticket text as the evidence. Do not invent facts.
Do not include Markdown fences or any text outside the JSON object.

EXAMPLES:
Example 1 — Ticket: "I was charged twice for the same order."
Answer: {"category": "billing", "reason": "Duplicate charge"}

Example 2 — Ticket: "My account has an order I never placed."
Answer: {"category": "fraud", "reason": "Unrecognized order"}

Example 3 — Ticket: "The app crashes whenever I open it."
Answer: {"category": "technical", "reason": "App crash"}

Example 4 — Ticket: "My delivery has not arrived."
Answer: {"category": "delivery", "reason": "Late delivery"}
""".strip()

assert "CLASSIFY" in ZERO_SHOT_PROMPT
assert "EXAMPLES:" not in ZERO_SHOT_PROMPT
assert "CLASSIFY" in FEW_SHOT_PROMPT
assert "EXAMPLES:" in FEW_SHOT_PROMPT


# ---------------------------------------------------------
# LOAD TICKETS AND REDACT PII
# ---------------------------------------------------------
tickets = pd.read_csv(
    RAW / "tickets.csv",
    dtype={
        "ticket_id": str,
        "customer_id": str,
        "order_id": str,
    },
)

tickets["text"] = tickets["text"].fillna("").astype(str)
tickets["customer_id"] = tickets["customer_id"].fillna("").astype(str)
tickets["ticket_id"] = tickets["ticket_id"].fillna("").astype(str)
tickets["true_category"] = (
    tickets["true_category"].fillna("").astype(str).str.lower().str.strip()
)

# Redact all 164 tickets, including empty and long tickets, for
# accurate substitution totals.
redacted_texts = []
redaction_counts = {name: 0 for name, _, _ in PII_PATTERNS}

for original_text in tickets["text"]:
    text = original_text

    for name, pattern, placeholder in PII_PATTERNS:
        text, count = pattern.subn(placeholder, text)
        redaction_counts[name] += count

    redacted_texts.append(text)

tickets["redacted_text"] = redacted_texts

print("\nPII REDACTION COUNTS")
for name, count in redaction_counts.items():
    print(f"{name}: {count}")
print("Total substitutions:", sum(redaction_counts.values()))

# Prove that no target PII pattern remains in any redacted ticket.
remaining_pii = 0

for text in tickets["redacted_text"]:
    for _, pattern, _ in PII_PATTERNS:
        remaining_pii += len(pattern.findall(text))

assert remaining_pii == 0, f"PII remains after redaction: {remaining_pii}"
print("PII matches remaining after redaction:", remaining_pii)
print("PII REDACTION ASSERTION PASSED")


# ---------------------------------------------------------
# LOAD CUSTOMER SEGMENTS FROM PART B
# ---------------------------------------------------------
segments_path = OUTPUTS / "customer_segments.csv"
segments_df = pd.read_csv(
    segments_path,
    dtype={"customer_id": str},
)

def to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}

segment_lookup = {}

for _, row in segments_df.iterrows():
    segment_lookup[str(row["customer_id"])] = {
        "segment_name": str(row.get("segment_name", "")),
        "is_premium": to_bool(row.get("is_premium", False)),
    }


# ---------------------------------------------------------
# RESPONSE VALIDATION
# ---------------------------------------------------------
def validate_response(content):
    """
    Return (status, category).

    Valid JSON with an unsupported category is invalid and must
    not fall through to keyword matching.
    """
    try:
        parsed = json.loads(content)

    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(content).lower())
        found = [word for word in words if word in ALLOWED]

        if len(found) == 1:
            return "classified_keyword", found[0]

        return "flagged_invalid_output", ""

    if isinstance(parsed, dict):
        category = parsed.get("category")
        if isinstance(category, str) and category.lower() in ALLOWED:
            return "classified_json", category.lower()

    return "flagged_invalid_output", ""


# ---------------------------------------------------------
# ROUTING
# ---------------------------------------------------------
HUMAN_REVIEW_STATUSES = {
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted",
}

def apply_routing(result):
    customer = segment_lookup.get(
        result["customer_id"],
        {"segment_name": "", "is_premium": False},
    )

    result["segment_name"] = customer["segment_name"]
    category = result["category"]

    if result["status"] in HUMAN_REVIEW_STATUSES:
        result["route"] = "human_review"
        result["priority"] = "NORMAL"

    elif category == "fraud":
        result["route"] = "security_team"
        result["priority"] = "HIGH"

    elif customer["is_premium"]:
        result["route"] = "premium_desk"
        result["priority"] = "HIGH"

    else:
        result["route"] = "support_queue"
        result["priority"] = "NORMAL"

    return result


# ---------------------------------------------------------
# FULL TRIAGE PIPELINE
# ---------------------------------------------------------
def run_pipeline(prompt, token_budget=None, run_name="run"):
    # Required: reset immediately before each independent full run.
    mock_llm.reset()

    running_tokens = 0
    results = []

    for _, ticket in tickets.iterrows():
        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        original_text = ticket["text"]
        redacted_text = ticket["redacted_text"]

        result = {
             "ticket_id": ticket_id,
            "customer_id": customer_id,
            "true_category": ticket["true_category"],
            "status": "",
            "category": "",
            "segment_name": "",
            "priority": "",
            "route": "",
            "tokens_used": 0,
        }

        # Input guardrails must happen before any model call.
        if not original_text.strip():
            result["status"] = "skipped_empty"
            results.append(apply_routing(result))
            continue

        if mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:
            result["status"] = "flagged_long"
            results.append(apply_routing(result))
            continue

        # Every prompt's user content must be exactly the redacted
        # ticket text, with no extra labels or instructions.
        assert redacted_text == ticket["redacted_text"]

        for _, pattern, _ in PII_PATTERNS:
            assert pattern.search(redacted_text) is None, (
                f"PII would reach model for ticket {ticket_id}"
            )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": redacted_text},
        ]

        response = None
        exhausted = False

        # Initial attempt + up to 3 retries.
        for retry_number in range(MAX_RETRIES + 1):
            # Check budget before EVERY model call, including retries.
            if (
                token_budget is not None
                and running_tokens >= token_budget
            ):
                exhausted = True
                break

            try:
                response = mock_llm.chat(messages)
                break

            except mock_llm.TransientLLMError:
                if retry_number >= MAX_RETRIES:
                    # Defensive fallback: never silently guess a category.
                    print(
                        f"ERROR: ticket {ticket_id} failed after "
                        f"{MAX_RETRIES} retries"
                    )
                    break

                wait = (
                    BASE_DELAY
                    * (2 ** retry_number)
                    * random.uniform(0.9, 1.15)
                )

                print(
                    f"RETRY ticket_id={ticket_id} "
                    f"retry_number={retry_number} wait={wait:.5f}s"
                )
                time.sleep(wait)

        if exhausted:
            result["status"] = "budget_exhausted"
            results.append(apply_routing(result))
            continue

        if response is None:
            result["status"] = "flagged_invalid_output"
            results.append(apply_routing(result))
            continue

        # Count tokens only when a response with usage is returned.
        used = int(response.usage.total_tokens)
        running_tokens += used
        result["tokens_used"] = used

        content = response.choices[0].message.content or ""
        status, category = validate_response(content)

        result["status"] = status
        result["category"] = category

        results.append(apply_routing(result))

    output = pd.DataFrame(results)

    print(f"\n{run_name.upper()} STATUS COUNTS")
    print(output["status"].value_counts().sort_index().to_string())
    print("Total response tokens:", running_tokens)
    print(
        "Classified tickets:",
        output["status"].isin(
            {"classified_json", "classified_keyword"}
        ).sum(),
    )
    print(
        "Budget-exhausted tickets:",
        (output["status"] == "budget_exhausted").sum(),
    )

    return output, running_tokens


# ---------------------------------------------------------
# ACCURACY AND CONFUSION TABLE
# ---------------------------------------------------------
def report_accuracy(output, run_name, total_tokens):
    classified = output[
        output["status"].isin({"classified_json", "classified_keyword"})
    ].copy()

    classified = classified[
        classified["true_category"].isin(ALLOWED)
    ]

    if len(classified):
        accuracy = (
            classified["category"] == classified["true_category"]
        ).mean()
    else:
        accuracy = 0.0

    print(f"\n{run_name.upper()} EVALUATION")
    print("Classified tickets:", len(classified))
    print(f"Accuracy over classified tickets: {accuracy:.4%}")
    print("Total response tokens:", total_tokens)

    if len(classified):
        print(
            "Tokens per classified ticket:",
            f"{total_tokens / len(classified):.2f}",
        )

    return classified


# ---------------------------------------------------------
# RUN 1: ZERO-SHOT, NO TOKEN BUDGET
# ---------------------------------------------------------
zero_results, zero_tokens = run_pipeline(
    ZERO_SHOT_PROMPT,
    token_budget=None,
    run_name="zero-shot",
)

zero_classified = report_accuracy(
    zero_results, "zero-shot", zero_tokens
)

print("\nZERO-SHOT CONFUSION TABLE")
zero_confusion = pd.crosstab(
    zero_classified["true_category"],
    zero_classified["category"],
).reindex(index=ALLOWED, columns=ALLOWED, fill_value=0)

print(zero_confusion.to_string())


# ---------------------------------------------------------
# RUN 2: FEW-SHOT, NO TOKEN BUDGET
# ---------------------------------------------------------
few_results, few_tokens = run_pipeline(
    FEW_SHOT_PROMPT,
    token_budget=None,
    run_name="few-shot",
)

few_classified = report_accuracy(
    few_results, "few-shot", few_tokens
)

print("\nPROMPT COST COMPARISON")
print("Zero-shot tokens:", zero_tokens)
print("Few-shot tokens:", few_tokens)
print("Extra few-shot tokens:", few_tokens - zero_tokens)

if len(zero_classified) and len(few_classified):
    zero_acc = (
        zero_classified["category"] == zero_classified["true_category"]
    ).mean()
    few_acc = (
        few_classified["category"] == few_classified["true_category"]
    ).mean()

    print(f"Zero-shot accuracy: {zero_acc:.4%}")
    print(f"Few-shot accuracy: {few_acc:.4%}")
    print(f"Accuracy improvement: {few_acc - zero_acc:.4%}")


# ---------------------------------------------------------
# RUN 3: ZERO-SHOT TOKEN-BUDGET DEMO
# ---------------------------------------------------------
budget_results, budget_tokens = run_pipeline(
    ZERO_SHOT_PROMPT,
    token_budget=TOKEN_BUDGET,
    run_name="zero-shot-budget-demo",
)


budget_path = OUTPUTS / "triage_budget_demo.csv"
budget_results.drop(columns=["true_category"]).to_csv(
    budget_path, index=False
)


budget_exhausted_count = (
    budget_results["status"] == "budget_exhausted"
).sum()

print("\nBUDGET DEMO")
print("Configured token budget:", TOKEN_BUDGET)
print("Budget-exhausted tickets:", budget_exhausted_count)
print("Saved:", budget_path)

assert budget_exhausted_count >= 1, (
    "No tickets exhausted the budget. Lower TOKEN_BUDGET and rerun."
)


# ---------------------------------------------------------
# FINAL OUTPUT: FEW-SHOT, LARGE TOKEN BUDGET
# ---------------------------------------------------------
final_results, final_tokens = run_pipeline(
    FEW_SHOT_PROMPT,
    token_budget=100000,
    run_name="final-few-shot",
)


final_path = OUTPUTS / "triage_results.csv"
final_results.drop(columns=["true_category"]).to_csv(
    final_path, index=False
)

print("\nFINAL OUTPUT")
print("Rows:", len(final_results))
print("Saved:", final_path)
print("Total response tokens:", final_tokens)

assert len(final_results) == 164, (
    f"Expected 164 tickets, found {len(final_results)}"
)

assert not final_results["status"].eq("").any()
print("PART C PIPELINE COMPLETED")
