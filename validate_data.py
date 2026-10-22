
import sqlite3
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
CLEAN = BASE / "data" / "clean"
DB = BASE / "data" / "instamart.db"

def check(condition, message):
    assert condition, f"FAILED: {message}"

def main():
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")

    # Check clean CSV row counts.
    check(len(customers) == 1200, "Customer count must be 1200")
    check(len(products) == 40, "Product count must be 40")
    check(len(orders) == 12309, "Order count must be 12309")

    # Check customer cleaning.
    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }
    check(
        customers["city"].value_counts().to_dict() == expected_cities,
        "Customer city counts do not match reference"
    )

    expected_genders = {
        "female": 610,
        "male": 560,
        "unknown": 30,
    }
    check(
        customers["gender"].value_counts().to_dict() == expected_genders,
        "Customer gender counts do not match reference"
    )

    check(customers["age"].notna().all(), "Customer ages contain missing values")
    check(
        customers["age"].between(18, 60).all(),
        "Customer ages must be between 18 and 60"
    )
    check(
        customers["signup_date"].notna().all(),
        "Signup dates contain missing values"
    )

    # Check order cleaning.
    check(orders["order_id"].is_unique, "Order IDs are not unique")
    check(
        orders["status"].value_counts().to_dict()
        == {"completed": 11110, "cancelled": 1199},
        "Order status counts do not match reference"
    )
    check(
        orders["order_amount"].notna().all(),
        "Order amounts contain missing values"
    )
    check(
        (orders["order_amount"] >= 0).all(),
        "Negative order amounts found"
    )
    check(
        orders.loc[
            orders["status"] == "completed", "delivery_minutes"
        ].notna().all(),
        "Completed orders have missing delivery times"
    )
    check(
        orders.loc[
            orders["status"] == "cancelled", "delivery_minutes"
        ].isna().all(),
        "Cancelled orders should have missing delivery times"
    )

    revenue = round(
        orders.loc[
            orders["status"] == "completed", "order_amount"
        ].sum(),
        2
    )
    check(
        abs(revenue - 2998668.80) < 0.01,
        f"Completed revenue mismatch: {revenue}"
    )

    # Check SQLite database integrity and table contents.
    with sqlite3.connect(DB) as con:
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        check(integrity == "ok", "SQLite integrity check failed")

        tables = {
            row[0]
            for row in con.execute("""
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """)
        }
        check(
            tables == {"customers", "products", "orders"},
            f"Unexpected database tables: {tables}"
        )

        for table, expected_count in {
            "customers": 1200,
            "products": 40,
            "orders": 12309,
        }.items():
            actual_count = con.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            check(
                actual_count == expected_count,
                f"{table} table row count mismatch"
            )

        orphan_customers = con.execute("""
            SELECT COUNT(*)
            FROM orders o
            LEFT JOIN customers c ON o.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
        """).fetchone()[0]

        orphan_products = con.execute("""
            SELECT COUNT(*)
            FROM orders o
            LEFT JOIN products p ON o.product_id = p.product_id
            WHERE p.product_id IS NULL
        """).fetchone()[0]

        check(orphan_customers == 0, "Orders reference missing customers")
        check(orphan_products == 0, "Orders reference missing products")

    print("ALL CHECKPOINTS PASSED")

if __name__ == "__main__":
    main()
