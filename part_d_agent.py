
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import sqlite3
import time
import uuid

import mock_llm
from part_d_rag import retrieve


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart.db"
SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Tool implementations ----------

def get_order(order_id):
    """Look up one order in the Part A SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": (
            None if row[3] is None else float(row[3])
        ),
    }


def search_policy(query):
    """Retrieve the three most relevant policy chunks."""
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Calculate a full refund only when delivery took more than 45 minutes."""
    amount = float(order_amount)
    minutes = (
        None if delivery_minutes is None else float(delivery_minutes)
    )

    refund = round(amount, 2) if minutes is not None and minutes > 45 else 0.0
    return {"refund_inr": refund}


def _write_ledger(tool_name, order_id, refund_inr):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"refund_inr": record["refund_inr"]}



def issue_refund(order_id, refund_inr):
    """Issue only a valid, completed-order refund of INR 1000 or less."""
    order = get_order(order_id)
    if "error" in order:
        return {"error": "order not found"}

    if order["status"] != "completed":
        return {"error": "only completed orders can be refunded"}

    expected = calculate_refund(
        order["order_amount"], order["delivery_minutes"]
    )["refund_inr"]
    amount = round(float(refund_inr), 2)

    if expected <= 0 or abs(amount - expected) > 0.01:
        return {"error": "refund amount does not satisfy policy"}
    if amount > 1000:
        return {"error": "refund above INR 1000 must be escalated"}

    return _write_ledger("issue_refund", order_id, amount)


def escalate_to_human(order_id, refund_inr):
    """Escalate a valid completed-order refund above INR 1000."""
    order = get_order(order_id)
    if "error" in order:
        return {"error": "order not found"}

    if order["status"] != "completed":
        return {"error": "only completed orders can be refunded"}

    expected = calculate_refund(
        order["order_amount"], order["delivery_minutes"]
    )["refund_inr"]
    amount = round(float(refund_inr), 2)

    if expected <= 1000 or abs(amount - expected) > 0.01:
        return {"error": "refund amount does not satisfy escalation policy"}

    return _write_ledger("escalate_to_human", order_id, amount)



TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order's status, amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve relevant policy chunks for a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the refund for an order's amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": ["number", "null"]},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record and issue an eligible automatic refund of INR 1000 or less.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund above INR 1000 for human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


# ---------- Checkpoint and result helpers ----------

class SimulatedCrash(RuntimeError):
    """Used only to demonstrate checkpoint recovery."""


def _checkpoint_path(run_id):
    return STATE_DIR / f"{run_id}.json"


def _save_checkpoint(state):
    path = _checkpoint_path(state["run_id"])
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _load_checkpoint(run_id):
    path = _checkpoint_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found for run_id={run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _status_from_answer(final_answer):
    if not final_answer:
        return "FALLBACK"
    return final_answer.split(":", 1)[0].strip()


def _fallback(status, detail):
    return f"{status}: Human handoff required. {detail}"


def _finish(state, status, final_answer):
    state["status"] = status
    state["final_answer"] = final_answer
    state["finished"] = True
    _save_checkpoint(state)

    return {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": status,
        "final_answer": final_answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "tools_executed": state["tools_executed"],
    }


def _new_state(user_message, scenario_id, run_id=None):
    run_id = run_id or f"{scenario_id}_{uuid.uuid4().hex[:8]}"

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "AGENT: You are an Instamart refund agent. Use the "
                    "provided tools to look up orders, retrieve policy, "
                    "calculate refunds, issue eligible refunds, or escalate "
                    "amounts above INR 1000. Only completed orders qualify. "
                    "Return a final answer beginning with REFUND, ESCALATE, "
                    "DENIED, or NOT_FOUND followed by a colon."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "steps": [],
        "tools_executed": [],
        "finished": False,
        "status": None,
        "final_answer": None,
    }


# ---------- ReAct execution loop ----------

def _continue_agent(
    state,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None,
):
    started = time.monotonic()

    if state.get("finished"):
        return _finish(
            state,
            state["status"],
            state["final_answer"],
        )

    # max_iterations is the total model-call ceiling for the entire run,
    # including calls made before a crash and calls made after resuming.
    while state["llm_calls"] < max_iterations:
        elapsed = time.monotonic() - started
        if elapsed >= time_limit_seconds:
            return _finish(
                state,
                "FALLBACK_TIMEOUT",
                _fallback(
                    "FALLBACK_TIMEOUT",
                    f"Time limit of {time_limit_seconds} seconds reached.",
                ),
            )

        try:
            response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        except mock_llm.TransientLLMError as exc:
            # The assignment's mock agent normally does not raise this,
            # but do not silently turn a transient failure into a refund.
            return _finish(
                state,
                "FALLBACK",
                _fallback("FALLBACK", f"Model call failed: {exc}"),
            )

        state["llm_calls"] += 1
        state["tokens_used"] += int(response.usage.total_tokens)

        # Check immediately after the call; do not execute a returned tool
        # if the call has pushed usage above the configured limit.
        if state["tokens_used"] > token_limit:
            return _finish(
                state,
                "FALLBACK_TOKEN_LIMIT",
                _fallback(
                    "FALLBACK_TOKEN_LIMIT",
                    f"Token limit of {token_limit} exceeded.",
                ),
            )

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            final_answer = message.content or "FALLBACK: Empty model response."
            status = _status_from_answer(final_answer)
            return _finish(state, status, final_answer)

        assistant_tool_calls = []
        parsed_calls = []

        for tool_call in tool_calls:
            name = tool_call.name
            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            assistant_tool_calls.append({
                "id": tool_call.id,
                "name": name,
                "arguments": arguments,
            })
            parsed_calls.append((tool_call.id, name, arguments))

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls,
        })

        for call_id, name, arguments in parsed_calls:
            if name not in TOOL_MAP:
                observation = {"error": f"unknown tool: {name}"}
            else:
                try:
                    observation = TOOL_MAP[name](**arguments)
                except (TypeError, ValueError, sqlite3.Error) as exc:
                    observation = {"error": str(exc)}

            state["step_number"] += 1
            step = {
                "step_number": state["step_number"],
                "tool_name": name,
                "arguments": arguments,
                "observation": observation,
            }
            state["steps"].append(step)
            state["tools_executed"].append(name)

            state["messages"].append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(
                    observation, ensure_ascii=False, default=str
                ),
            })

            # Persist the observation and full conversation before any
            # simulated crash can happen.
            _save_checkpoint(state)

            if (
                crash_after_tools is not None
                and len(state["tools_executed"]) == crash_after_tools
            ):
                raise SimulatedCrash(
                    f"Simulated crash after tool execution {crash_after_tools}"
                )

    return _finish(
        state,
        "FALLBACK_MAX_ITERATIONS",
        _fallback(
            "FALLBACK_MAX_ITERATIONS",
            f"Reached the limit of {max_iterations} model calls without a final answer.",
        ),
    )


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
):
    """Start a fresh ReAct run and checkpoint after each tool observation."""
    state = _new_state(user_message, scenario_id)
    _save_checkpoint(state)

    return _continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
    )


def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
):
    """Resume from the latest checkpoint without repeating saved tool steps."""
    state = _load_checkpoint(run_id)

    return _continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
    )


# ---------- Scenario evaluation and demonstrations ----------

def load_scenarios():
    import csv

    with SCENARIOS_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_all_scenarios():
    """Run the seven supplied scenarios and write one JSONL record per run."""
    scenarios = load_scenarios()
    results = []

    # Keep prior records if rerunning, but this run appends exactly one
    # new record for each scenario.
    for scenario in scenarios:
        result = run_agent(
            scenario["message"],
            scenario["scenario_id"],
        )

        order_match = re.search(r"ORD-\d{6}", scenario["message"])
        order = get_order(order_match.group(0)) if order_match else {}
        expected_status = scenario["expected_outcome"]
        expected_amount = float(scenario["expected_refund_inr"])

        actual_amount = 0.0
        if order and "error" not in order:
            # The scenario's expected amount is for checking only.
            # Actual refund actions are grounded in tool observations.
            for step in result["steps"]:
                if step["tool_name"] in {
                    "issue_refund", "escalate_to_human"
                }:
                    actual_amount = float(
                        step["observation"].get("refund_inr", 0.0)
                    )

        record = {
            "scenario_id": scenario["scenario_id"],
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"],
        }
        append_jsonl(RUNS_PATH, record)

        results.append({
            "scenario_id": scenario["scenario_id"],
            "expected_status": expected_status,
            "status": result["status"],
            "expected_refund_inr": expected_amount,
            "refund_inr": actual_amount,
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "result": result,
        })

    print("\n--- SEVEN SCENARIOS ---")
    print(
        "scenario | expected | actual | expected INR | actual INR | calls | tokens"
    )
    for item in results:
        print(
            f"{item['scenario_id']:8} | "
            f"{item['expected_status']:8} | "
            f"{item['status']:22} | "
            f"{item['expected_refund_inr']:12.2f} | "
            f"{item['refund_inr']:10.2f} | "
            f"{item['llm_calls']:5} | "
            f"{item['tokens_used']}"
        )

    return results


def run_safety_demos():
    """Demonstrate the token and timeout fallbacks required by the brief."""
    s1 = next(
        row for row in load_scenarios() if row["scenario_id"] == "S1"
    )

    token_demo = run_agent(
        s1["message"],
        "S1_TOKEN_LIMIT_DEMO",
        token_limit=200,
    )
    timeout_demo = run_agent(
        s1["message"],
        "S1_TIMEOUT_DEMO",
        time_limit_seconds=0,
    )

    print("\n--- SAFETY DEMONSTRATIONS ---")
    print(
        "S1 token_limit=200:",
        token_demo["status"],
        "| calls:", token_demo["llm_calls"],
        "| tokens:", token_demo["tokens_used"],
    )
    print(
        "S1 time_limit_seconds=0:",
        timeout_demo["status"],
        "| calls:", timeout_demo["llm_calls"],
        "| tokens:", timeout_demo["tokens_used"],
    )

    return token_demo, timeout_demo


def run_crash_resume_demo():
    """Crash after tool 2, then resume without repeating prior tool actions."""
    s1 = next(
        row for row in load_scenarios() if row["scenario_id"] == "S1"
    )
    run_id = "S1_CRASH_RESUME_DEMO"

    # Start from a fresh checkpoint for a repeatable demonstration.
    checkpoint = _checkpoint_path(run_id)
    if checkpoint.exists():
        checkpoint.unlink()

    state = _new_state(s1["message"], s1["scenario_id"], run_id=run_id)
    _save_checkpoint(state)

    try:
        _continue_agent(
            state,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
            crash_after_tools=2,
        )
    except SimulatedCrash as exc:
        print("\nSimulated crash:", exc)

    checkpoint_state = _load_checkpoint(run_id)
    tools_before_resume = list(checkpoint_state["tools_executed"])

    resumed = resume_agent(
        run_id,
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10,
    )

    combined_tools = resumed["tools_executed"]
    print("Tools before resume:", tools_before_resume)
    print("Tools across crash + resume:", combined_tools)
    print("Final status:", resumed["status"])
    print("Final answer:", resumed["final_answer"])
    print("Model calls across both parts:", resumed["llm_calls"])

    expected_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
    ]
    assert combined_tools == expected_tools, (
        f"Unexpected tool execution sequence: {combined_tools}"
    )
    assert resumed["status"] == "REFUND"
    assert abs(
        float(re.search(r"INR\s+([\d.]+)", resumed["final_answer"]).group(1))
        - 128.25
    ) < 0.01

    return resumed


if __name__ == "__main__":
    results = run_all_scenarios()
    token_demo, timeout_demo = run_safety_demos()
    crash_demo = run_crash_resume_demo()

    print("\n--- REQUIRED CHECKS ---")
    for item in results:
        expected_status = item["expected_status"]
        actual_status = item["status"]

        if expected_status == "FALLBACK":
            assert actual_status == "FALLBACK_MAX_ITERATIONS", (
                f"{item['scenario_id']} expected max-iteration fallback, "
                f"got {actual_status}"
            )
        else:
            assert actual_status == expected_status, (
                f"{item['scenario_id']}: expected {expected_status}, "
                f"got {actual_status}"
            )

        assert abs(
            item["refund_inr"] - item["expected_refund_inr"]
        ) < 0.01, (
            f"{item['scenario_id']}: refund amount mismatch "
            f"({item['refund_inr']} vs {item['expected_refund_inr']})"
        )

    assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
    assert timeout_demo["status"] == "FALLBACK_TIMEOUT"
    assert crash_demo["tools_executed"] == [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
    ]

    print("All Part D agent checks passed.")
