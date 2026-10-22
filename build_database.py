
import sqlite3
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
CLEAN = BASE / "data" / "clean"
DB = BASE / "data" / "instamart.db"

def read_csv(name):
    return pd.read_csv(RAW / name)

def strip_text(df):
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(
            lambda x: x.strip() if isinstance(x, str) else x
        )
    return df

def main():
    CLEAN.mkdir(parents=True, exist_ok=True)

    # 1. Clean customers.
    customers = strip_text(read_csv("customers.csv"))
    customers = customers.drop_duplicates("customer_id").copy()

    customers["city"] = customers["city"].str.strip().str.title()

    gender = customers["gender"].fillna("").str.strip().str.lower()
    customers["gender"] = gender.map({
        "male": "male", "m": "male",
        "female": "female", "f": "female"
    }).fillna("unknown")

    customers["age"] = pd.to_numeric(
        customers["age"], errors="coerce"
    )
    customers.loc[
        ~customers["age"].between(10, 100), "age"
    ] = float("nan")

    customers["age"] = customers["age"].fillna(
        customers.groupby("city")["age"].transform("median")
    )
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # 2. Clean products.
    products = strip_text(read_csv("products.csv"))
    products = products.drop_duplicates("product_id").copy()
    products["price_inr"] = pd.to_numeric(
        products["price_inr"], errors="coerce"
    )

    # 3. Clean orders in the required sequence.
    orders = strip_text(read_csv("orders.csv"))
    raw_count = len(orders)

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    exact_duplicates = int(orders.duplicated().sum())
    orders = orders.drop_duplicates().copy()

    # Keep the latest status record for each order.
    orders = orders.sort_values(
        ["order_id", "order_ts"], kind="mergesort"
    )
    orders = orders.drop_duplicates(
        subset=["order_id"], keep="last"
    ).copy()

    for col in [
        "quantity", "discount_pct", "order_amount", "delivery_minutes"
    ]:
        orders[col] = pd.to_numeric(orders[col], errors="coerce")

    # Recover missing/unparseable amounts from catalogue prices.
    orders = orders.merge(
        products[["product_id", "category", "price_inr"]],
        on="product_id", how="left", validate="many_to_one"
    )

    orders["order_amount"] = orders["order_amount"].fillna(
        (
            orders["price_inr"]
            * orders["quantity"]
            * (1 - orders["discount_pct"] / 100)
        ).round(2)
    )

    # Negative delivery durations are impossible.
    orders.loc[
        orders["delivery_minutes"] < 0, "delivery_minutes"
    ] = float("nan")

    # Impute missing delivery times only for completed orders,
    # using the completed-order median for that product category.
    completed = orders["status"].eq("completed")
    medians = orders.loc[completed].groupby(
        "category"
    )["delivery_minutes"].transform("median")

    completed_index = orders.index[completed]
    orders.loc[completed_index, "delivery_minutes"] = (
        orders.loc[completed_index, "delivery_minutes"].fillna(medians)
    )

    # Store timestamps as YYYY-MM-DD HH:MM:SS text.
    orders["order_ts"] = orders["order_ts"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Remove helper column; only the required order columns remain.
    orders = orders[
        [
            "order_id", "customer_id", "product_id", "quantity",
            "order_ts", "status", "discount_pct", "order_amount",
            "delivery_minutes"
        ]
    ].copy()

    # Write clean CSVs.
    customers.to_csv(CLEAN / "customers.csv", index=False)
    products.to_csv(CLEAN / "products.csv", index=False)
    orders.to_csv(CLEAN / "orders.csv", index=False)

    # Create the required SQLite database with exactly three tables.
    if DB.exists():
        DB.unlink()

    with sqlite3.connect(DB) as con:
        customers.to_sql("customers", con, index=False)
        products.to_sql("products", con, index=False)
        orders.to_sql("orders", con, index=False)

        con.executescript("""
            CREATE INDEX idx_orders_customer ON orders(customer_id);
            CREATE INDEX idx_orders_product ON orders(product_id);
            CREATE INDEX idx_orders_timestamp ON orders(order_ts);
        """)

        con.execute("PRAGMA foreign_keys = ON")

        # Validation output.
        completed_revenue = con.execute("""
            SELECT ROUND(SUM(order_amount), 2)
            FROM orders WHERE status = 'completed'
        """).fetchone()[0]

        counts = {
            table: con.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            for table in ["customers", "products", "orders"]
        }

    print(f"Raw order rows: {raw_count}")
    print(f"Exact duplicate order rows removed: {exact_duplicates}")
    print(f"Clean customers: {counts['customers']}")
    print(f"Clean products: {counts['products']}")
    print(f"Clean orders: {counts['orders']}")
    print("Order statuses:")
    print(orders["status"].value_counts().sort_index().to_string())
    print(
        "Missing completed delivery times:",
        int(orders.loc[orders["status"].eq("completed"),
                       "delivery_minutes"].isna().sum())
    )
    print(
        "Missing cancelled delivery times:",
        int(orders.loc[orders["status"].eq("cancelled"),
                       "delivery_minutes"].isna().sum())
    )
    print(f"Completed revenue: INR {completed_revenue:.2f}")
    print(f"Database: {DB}")

if __name__ == "__main__":
    main()
