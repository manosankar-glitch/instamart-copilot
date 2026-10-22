
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

BASE = Path(__file__).resolve().parent
CLEAN = BASE / "data" / "clean"
OUTPUTS = BASE / "outputs"

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    products = pd.read_csv(CLEAN / "products.csv")

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    # Attach product category for historical favourite-category features.
    orders = orders.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    completed = orders.loc[
        orders["status"].eq("completed")
    ].copy()

    # Define the cohort using completed orders strictly before cutoff.
    history_completed = completed.loc[
        completed["order_ts"] < CUTOFF
    ].copy()

    cohort_ids = history_completed["customer_id"].unique()

    # Label: no completed order in September 2025 means lapsed=1.
    september_completed = completed.loc[
        (completed["order_ts"] >= CUTOFF)
        & (completed["order_ts"] < LABEL_END)
    ]

    ordered_in_september = set(
        september_completed["customer_id"].unique()
    )

    labels = pd.DataFrame({"customer_id": cohort_ids})
    labels["lapsed"] = (
        ~labels["customer_id"].isin(ordered_in_september)
    ).astype(int)

    # All model features use only records strictly before cutoff.
    history_orders = orders.loc[
        orders["order_ts"] < CUTOFF
    ].copy()

    history_completed = history_orders.loc[
        history_orders["status"].eq("completed")
    ].copy()

    # Completed-order behaviour features.
    features = (
        history_completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_delivery_minutes=("delivery_minutes", "mean"),
        )
        .reset_index()
    )

    features["recency"] = (
        CUTOFF - features["last_order_ts"]
    ).dt.days

    features = features.drop(columns=["last_order_ts"])

    # Cancellation rate uses all final-status orders before cutoff.
    status_counts = (
        history_orders.groupby("customer_id")
        .agg(
            all_final_orders=("order_id", "nunique"),
            cancelled_orders=(
                "status",
                lambda values: int(values.eq("cancelled").sum()),
            ),
        )
        .reset_index()
    )

    status_counts["cancellation_rate"] = (
        status_counts["cancelled_orders"]
        / status_counts["all_final_orders"]
    )

    # Most frequently ordered category among completed historical orders.
    def favourite_category(group):
        counts = group["category"].value_counts()
        # Deterministic tie-break: alphabetical category name.
        return sorted(counts[counts == counts.max()].index)[0]

    favourite = (
        history_completed.groupby("customer_id")
        .apply(favourite_category, include_groups=False)
        .rename("favourite_category")
        .reset_index()
    )

    # Customer demographics and tenure at the cutoff.
    customer_features = customers[
        ["customer_id", "city", "age", "gender", "signup_date"]
    ].copy()

    customer_features["tenure_days"] = (
        CUTOFF - customer_features["signup_date"]
    ).dt.days

    customer_features = customer_features.drop(
        columns=["signup_date"]
    )

    # Assemble the model table. No segment labels are joined.
    dataset = (
        labels
        .merge(features, on="customer_id", how="left", validate="one_to_one")
        .merge(status_counts, on="customer_id", how="left",
               validate="one_to_one")
        .merge(favourite, on="customer_id", how="left",
               validate="one_to_one")
        .merge(customer_features, on="customer_id", how="left",
               validate="one_to_one")
    )

    dataset = dataset.drop(columns=["all_final_orders", "cancelled_orders"])

    required_features = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "favourite_category",
        "city",
        "age",
        "gender",
        "tenure_days",
    ]

    missing_counts = dataset[required_features].isna().sum()
    print("\nMISSING FEATURE VALUES")
    print(missing_counts.to_string())

    if dataset[required_features].isna().any().any():
        raise ValueError(
            "Missing feature values found. Inspect the counts above before training."
        )

    dataset.to_csv(OUTPUTS / "lapse_features.csv", index=False)

    # Reproducible stratified split.
    X = dataset[required_features]
    y = dataset["lapsed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Baseline: always predict that the customer will not lapse.
    baseline_predictions = [0] * len(y_test)

    print("\nCOHORT SUMMARY")
    print("Cohort customers:", len(dataset))
    print("Lapsed customers:", int(y.sum()))
    print("Lapsed percentage:", round(100 * y.mean(), 2))
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))

    print("\nBASELINE — ALWAYS PREDICT NOT LAPSED")
    print(
        "Accuracy:",
        round(accuracy_score(y_test, baseline_predictions), 4),
    )
    print(
        "Lapsed-class F1:",
        round(f1_score(y_test, baseline_predictions, zero_division=0), 4),
    )

    print("\nSaved: outputs/lapse_features.csv")
    print("Features use only orders before 2025-09-01.")
    print("September completed orders are used only to create the label.")
    print("Task 3 segment labels are not used as model features.")


if __name__ == "__main__":
    main()
