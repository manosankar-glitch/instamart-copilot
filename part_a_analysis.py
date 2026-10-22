
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
CLEAN = BASE / "data" / "clean"
REPORTS = BASE / "reports"

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    raw_orders = pd.read_csv(RAW / "orders.csv")
    raw_customers = pd.read_csv(RAW / "customers.csv")
    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    products = pd.read_csv(CLEAN / "products.csv")

    # Raw-data quality audit.
    raw_amount = pd.to_numeric(
        raw_orders["order_amount"], errors="coerce"
    )
    raw_delivery = pd.to_numeric(
        raw_orders["delivery_minutes"], errors="coerce"
    )
    raw_age = pd.to_numeric(
        raw_customers["age"], errors="coerce"
    )

    audit = {
        "Raw order rows": len(raw_orders),
        "Exact duplicate order rows": int(raw_orders.duplicated().sum()),
        "Pending raw order rows": int(
            raw_orders["status"].astype(str).str.strip().str.lower()
            .eq("pending").sum()
        ),
        "Distinct raw order IDs": int(raw_orders["order_id"].nunique()),
        "Nonnumeric or missing order amounts": int(raw_amount.isna().sum()),
        "Negative delivery durations": int((raw_delivery < 0).sum()),
        "Blank delivery durations": int(
            raw_orders["delivery_minutes"].isna().sum()
        ),
        "Blank customer ages": int(raw_customers["age"].isna().sum()),
        "Customer ages outside 10-100": int(
             ((raw_age < 10) | (raw_age > 100)).sum()
        ),
        "Distinct raw city spellings": int(
            raw_customers["city"].nunique()
        ),
        "Distinct nonblank raw gender spellings": int(
            raw_customers.loc[
                raw_customers["gender"].notna()
                & raw_customers["gender"].astype(str).str.strip().ne(""),
                "gender"
            ].nunique()
        ),
        "Blank customer genders": int(
            raw_customers["gender"].isna().sum()
            + raw_customers["gender"].astype(str).str.strip().eq("").sum()
        ),
    }

    # Clean-data summary.
    completed = orders["status"].eq("completed")
    cancelled = orders["status"].eq("cancelled")

    completed_revenue = round(
        orders.loc[completed, "order_amount"].sum(), 2
    )

    summary = {
        "Clean customers": len(customers),
        "Clean products": len(products),
        "Clean orders": len(orders),
        "Completed orders": int(completed.sum()),
        "Cancelled orders": int(cancelled.sum()),
        "Completed revenue (INR)": f"{completed_revenue:.2f}",
        "Missing order amounts": int(orders["order_amount"].isna().sum()),
        "Completed orders missing delivery time": int(
            orders.loc[completed, "delivery_minutes"].isna().sum()
        ),
        "Cancelled orders missing delivery time": int(
            orders.loc[cancelled, "delivery_minutes"].isna().sum()
        ),
    }

    lines = [
        "# Part A — Data Audit, Cleaning and SQL Analysis",
        "",
        "## Raw data quality audit",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in audit.items())

    lines.extend([
        "",
        "## Clean dataset validation summary",
        "",
        "| Metric | Result |",
        "|---|---:|",
    ])
    lines.extend(f"| {key} | {value} |" for key, value in summary.items())

    lines.extend([
        "",
        "## Cleaning decisions",
        "",
        "- Removed exact duplicate order rows.",
        "- Converted order timestamps and numeric fields to usable types.",
        "- Kept the latest timestamped record for each order ID.",
        "- Normalized city names and mapped gender values to male, female, or unknown.",
        "- Treated customer ages outside 10–100 as missing and filled missing ages using city medians.",
        "- Recovered missing order amounts using product price, quantity, and discount.",
        "- Treated negative delivery durations as missing.",
        "- Filled missing delivery times for completed orders using the completed-order median for each product category.",
        "- Kept delivery durations missing for cancelled orders.",
        "",
        "## SQL findings",
        "",
        "- Completed-order revenue was INR 2,998,668.80 across 11,110 completed orders.",
        "- Electronics had the highest completed-order revenue among product categories.",
        "- Mumbai had the highest cancellation rate among the six cities.",
        "- Personal Care had the highest average delivery time among the listed product categories.",
        "- Monthly completed-order revenue varied across April–September 2025.",
        "",
        "These are descriptive findings from the supplied dataset; they do not establish why revenue or cancellation patterns differ.",
        "",
        "## Reproduction",
        "",
        "Run `python build_database.py`, then `python validate_data.py`, then `python run_sql_queries.py`, and finally `python part_a_analysis.py`.",
    ])

    report_path = REPORTS / "part_a_summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("RAW DATA AUDIT")
    for key, value in audit.items():
        print(f"{key}: {value}")

    print("\nCLEAN DATA SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nReport saved: {report_path.relative_to(BASE)}")

if __name__ == "__main__":
    main()
