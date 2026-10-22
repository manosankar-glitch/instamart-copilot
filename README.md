# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds an offline customer intelligence and support workflow using Python, SQLite, pandas, NumPy, scikit-learn, and the supplied `mock_llm.py`.

The project covers four parts:

- **Part A:** Data cleaning, auditing, SQLite database construction, SQL analysis, and validation.
- **Part B:** Customer RFM segmentation, lapse prediction, model evaluation, and retention prioritisation.
- **Part C:** Privacy-aware support-ticket classification, retry handling, budget controls, and segment-based routing.
- **Part D:** TF-IDF policy retrieval, grounded policy answers, refund tools, a ReAct agent, safety limits, and checkpoint-based recovery.

All model interactions use the supplied offline mock. No API key, paid model, or external model service is required.

## Reproduce

### Requirements

- Python 3.11 or later
- SQLite 3.25 or later
- Windows, macOS, or Linux
- Dependencies listed in `requirements.txt`

### Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/manosankar-glitch/instamart-copilot.git
cd instamart-copilot
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Generate Data and Run the Project

Run all commands from the repository root, in the following order:

```bash
python generate_data.py
python build_database.py
python validate_data.py
python run_sql_queries.py
python part_a_analysis.py
python part_b_segmentation.py
python part_b_lapse.py
python part_b_train.py
python part_c_triage.py
python part_d_rag.py
python part_d_agent.py
```

The `generate_data.py` script creates the synthetic raw datasets and policy files required by the workflow. The supplied `generate_data.py` and `mock_llm.py` files remain unmodified.

The validation script should report `ALL CHECKPOINTS PASSED` when the Part A validation checks succeed.

### Key Outputs

- `data/instamart.db`
- `outputs/customer_rfm.csv`
- `outputs/customer_segments.csv`
- `outputs/retention_list.csv`
- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `reports/sql/`
- `state/` checkpoint files

All model interactions use the supplied offline mock. No API key, paid model, or external model service is required.

## Part A — Data Cleaning, Database and SQL Analysis

### Data quality and cleaning

The raw order dataset contains 13,084 rows. The cleaning pipeline handles duplicate rows, pending orders, invalid or missing order amounts, delivery-time anomalies, missing customer ages, and inconsistent demographic values.

The cleaned SQLite database contains exactly three tables: `customers`, `products`, and `orders`.

| Metric | Result |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Cleaned orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Completed-order revenue | INR 2,998,668.80 |
| Missing completed-order delivery times | 0 |
| Missing cancelled-order delivery times | 1,199 |

The seven SQL queries are maintained in `sql/queries.sql`. Their result files are saved as `reports/sql/q1_results.csv` through `reports/sql/q7_results.csv`.

The data validation script checks the required data integrity conditions and reports `ALL CHECKPOINTS PASSED`.

## Part B — Customer Segmentation and Lapse Prediction

### RFM segmentation

RFM features are calculated for customers using recency, frequency, and monetary value, with 1 October 2025 as the reference date.

The RFM dataset contains 1,175 customers.

| RFM metric | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| Recency (days) | 33.56 | 0 | 180 |
| Frequency (orders) | 9.46 | — | 33 |
| Monetary value (INR) | 2,552.06 | — | 14,766.55 |

KMeans models with 2 through 8 clusters were evaluated using inertia and silhouette score.

| Clusters | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1,820.761 | 0.465 |
| 3 | 1,268.235 | 0.445 |
| 4 | 743.071 | 0.526 |
| 5 | 617.367 | 0.480 |
| 6 | 503.171 | 0.482 |
| 7 | 408.460 | 0.416 |
| 8 | 366.503 | 0.412 |

Four clusters were selected. Their profiles are:

| Segment | Customers | Mean recency | Mean frequency | Mean monetary value (INR) |
|---|---:|---:|---:|---:|
| Developing Customers | 436 | 28.86 | 3.77 | 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 |
| At-Risk Occasional | 182 | 109.04 | 2.20 | 685.99 |

The first two principal components explain approximately 85.23% of the variance.

### Lapse prediction

The lapse cohort contains 1,156 customers, of whom 443 (38.32%) are labelled lapsed. The train/test split contains 924 training observations and 232 test observations.

Logistic Regression and Random Forest were tuned using cross-validation. The selected Random Forest model achieved a test F1 score of approximately 0.655.

| Model | Test accuracy | Precision | Recall | F1 score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7284 | 0.6444 | 0.6517 | 0.6480 |
| Random Forest | 0.7371 | 0.6591 | 0.6517 | 0.6554 |
| Oversampled Logistic Regression | — | 0.5891 | 0.8539 | 0.6972 |

The Random Forest feature importance ranking is led by frequency, monetary value, and recency. Model persistence and reload prediction consistency are checked using joblib.

The retention list is saved in `outputs/retention_list.csv`. Its premium label is illustrative because the segmentation flag uses the full-window RFM data; it should not be interpreted as a leakage-free evaluation feature.

## Part C — Support-Ticket Triage

### Privacy and input validation

Ticket text is redacted before being sent to the mock model.

| Pattern | Redactions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total** | **278** |

The post-redaction assertions confirm that no matching email, phone, or order-ID patterns remain in the redacted ticket text.

Empty tickets are skipped, and long tickets are flagged for human review. Model failures are retried with exponential backoff and jitter, up to three retries. Invalid output is handled with JSON validation and a controlled keyword fallback.

### Classification evaluation

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| JSON classifications | 135 | 135 |
| Keyword-fallback classifications | 12 | 12 |
| Invalid-output flags | 9 | 9 |
| Long-ticket flags | 4 | 4 |
| Empty-ticket skips | 4 | 4 |
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy on classified tickets | 85.71% | 100.00% |
| Total response tokens | 16,264 | 30,304 |
| Tokens per classified ticket | 110.64 | 206.15 |

Few-shot prompting improved accuracy by 14.29 percentage points in this deterministic evaluation, at the cost of 14,040 additional response tokens.

The zero-shot confusion matrix for classified tickets is:

| Actual category | Predicted billing | Predicted delivery | Predicted technical | Predicted fraud |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

### Budget and routing

The zero-shot budget demonstration uses a token budget of 6,000. It classified 55 tickets, flagged 3 invalid outputs, skipped 4 empty tickets, flagged 4 long tickets, and marked 98 tickets as `budget_exhausted`.

The final few-shot output uses a budget of 100,000 and is saved to `outputs/triage_results.csv` with 164 rows.

Routing rules send fraud classifications to `security_team` with high priority, premium customers to `premium_desk` with high priority, and other valid classifications to `support_queue` with normal priority. Empty, long, invalid-output, and budget-exhausted tickets are sent to `human_review`.

The classification status records the processing outcome, not a guarantee that the predicted category is factually correct. Human review remains necessary for flagged cases.

## Part D — Policy RAG Assistant and Refund Agent

### 1. Policy chunking and retrieval

The five policy documents are split into 40-word chunks with a 10-word overlap, producing 15 chunks.

| Policy document | Chunks |
|---|---:|
| `refund_policy.txt` | 4 |
| `delivery_policy.txt` | 3 |
| `escalation_policy.txt` | 3 |
| `payment_policy.txt` | 3 |
| `privacy_policy.txt` | 2 |
| **Total** | **15** |

A `TfidfVectorizer()` is fitted on the chunk texts. Queries are transformed using the same fitted vectorizer, and chunks are ranked using cosine similarity. For the query `refund policy late delivery`, all three top-ranked chunks came from `refund_policy.txt`.

Term frequency (TF) measures how often a term appears in a chunk, while inverse document frequency (IDF) reduces the weight of terms that occur across many chunks. A limitation of TF-IDF is that it can miss semantically equivalent phrases that use different words.

### 2. Retrieval and grounding evaluation

| Evaluation | 10-word overlap | Zero overlap |
|---|---:|---:|
| Expected document in top three, answerable questions | 12/12 | 12/12 |
| Expected fact in answer, answerable questions | 12/12 | 10/12 |
| Correct fallback, unanswerable questions | 2/2 | 2/2 |

With zero overlap, the question about the full-refund delivery threshold lost the phrase `more than 45 minutes` across a chunk boundary. The payment-failure question also lost the expected `48 hours` fact. Overlapping chunks preserve context near chunk boundaries.

The policy assistant sends a system message marked `POLICY ASSISTANT` and a user message containing the retrieved context and question. It returns the model's answer and the sources used.

### 3. Refund rules and tools

The agent exposes five tools through `TOOLS_SPEC` and `TOOL_MAP`:

- `get_order` retrieves order status, amount, and delivery time from SQLite.
- `search_policy` retrieves the most relevant policy chunks.
- `calculate_refund` computes the eligible refund amount.
- `issue_refund` records an automatic refund.
- `escalate_to_human` records a refund requiring human approval.

The implemented rules are:

1. Only completed orders can be refunded.
2. Delivery taking more than 45 minutes qualifies for a full refund of the order amount.
3. Delivery taking 45 minutes or less receives no late-delivery refund.
4. Refunds above INR 1,000 are escalated rather than issued automatically.
5. All simulated requests are treated as falling within the policy's seven-day window.

The ledger-writing tools validate the order status and refund amount before recording a simulated action.

### 4. Seven-scenario agent evaluation

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1,142 |
| S2 | REFUND | 135.00 | 5 | 1,142 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,129 |
| S4 | DENIED | 0.00 | 4 | 801 |
| S5 | DENIED | 0.00 | 3 | 484 |
| S6 | NOT_FOUND | 0.00 | 2 | 170 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,294 |

The scenario results match the expected outcomes and refund amounts. S7 reaches the maximum of six model calls without a final answer and safely returns `FALLBACK_MAX_ITERATIONS`.

### 5. Safety demonstrations

| Test | Configuration | Result | Model calls | Tokens used |
|---|---|---|---:|---:|
| S1 token-limit demo | `token_limit=200` | FALLBACK_TOKEN_LIMIT | 3 | 501 |
| S1 timeout demo | `time_limit_seconds=0` | FALLBACK_TIMEOUT | 0 | 0 |

The token-limit fallback occurs immediately after a model call pushes cumulative usage above the limit, before the requested tool is executed. The zero-second timeout stops the agent before its first model call.

### 6. Checkpoint and resume demonstration

A simulated crash was triggered after the second tool execution, after the observation was saved to a checkpoint. Resuming the run completed the workflow without repeating either earlier tool.

Tool execution order across the crash and resume:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

The resumed run returned `REFUND: INR 128.25 issued.` The checkpoint contains the run ID, scenario ID, full message history, token usage, step number, and executed tools.

### 7. ReAct versus plan-and-execute

The ReAct agent made five model calls for S1, three for S5, and two for S6. It uses additional model calls and tokens because it observes each tool result before choosing the next action. A plan-and-execute design could request the complete action sequence in one model call, potentially reducing token usage and latency for straightforward cases.

However, a fixed plan can become unsafe when a tool returns an unexpected result, such as a missing order or a cancelled order. It must validate observations and replan on errors before issuing a refund. For this workflow, I prefer ReAct because each decision can be grounded in the actual order status, delivery time, calculated refund, and escalation threshold. The extra calls are justified by the financial safeguards and explicit audit trail.

### 8. Outputs and limitations

- `outputs/agent_runs.jsonl` — scenario outcomes, model calls, token usage, and tool observations.
- `outputs/refund_ledger.jsonl` — simulated refund issuance and escalation records.
- `state/` — run checkpoints used for resumption.
- `outputs/triage_results.csv` — Part C ticket classifications and routing.
- `outputs/triage_budget_demo.csv` — Part C token-budget demonstration.

The supplied mock model is deterministic and offline, so these results do not establish production-level model accuracy. TF-IDF retrieval is sensitive to vocabulary and chunk boundaries. Real deployment would require stronger policy-grounding checks, authentication and authorization, idempotent refund operations, monitoring, and human oversight for uncertain or high-value decisions.

All refunds and escalations in this project are simulated. No real money is moved.
## Limitations and Next Steps

### Limitations
- The project uses generated and synthetic data, so results may differ from real-world Instamart customer and support-ticket data.
- The language model is mocked for reproducible, no-cost evaluation; results do not establish the performance of a production LLM.
- Classification, customer segmentation, retrieval, and refund-agent results depend on the supplied dataset, rules, and evaluation scenarios.
- The refund workflow is a demonstration. Production deployment would require authenticated services, transaction verification, stronger access controls, audit logging, and human escalation procedures.

### Next Steps
- Evaluate the pipeline on approved, anonymized real-world data.
- Test classification and retrieval across more diverse and challenging examples.
- Add monitoring for accuracy, privacy protection, policy-grounding errors, and agent tool failures.
- Integrate authenticated production services only after security, compliance, and refund-authorization reviews.
- Document reproducible evaluation results and rerun all validation checks after changes.
