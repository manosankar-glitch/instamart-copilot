
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
CLEAN = BASE / "data" / "clean"
OUTPUTS = BASE / "outputs"
REPORTS = BASE / "reports"

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    # RFM uses completed orders only.
    completed = orders.loc[
        orders["status"].eq("completed")
    ].copy()

    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (
        SNAPSHOT - rfm["last_order_ts"]
    ).dt.days

    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ].copy()

    # Print and save RFM reference checks.
    print("\nRFM SUMMARY")
    print(rfm[["recency", "frequency", "monetary"]].describe().round(2))
    print("Customers with completed orders:", len(rfm))

    rfm.to_csv(OUTPUTS / "customer_rfm.csv", index=False)

    # Scale RFM once for reproducible clustering comparisons.
    features = ["recency", "frequency", "monetary"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[features])

    # Evaluate every K requested by the assignment.
    metric_rows = []

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        )
        labels = model.fit_predict(scaled)

        metric_rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(scaled, labels),
        })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUTS / "kmeans_metrics.csv", index=False)

    print("\nK-MEANS METRICS")
    print(metrics.round(3).to_string(index=False))

    # Final K chosen from the requested {3, 4, 5} candidates.
    # K=4 is the reference choice and gives actionable segment profiles.
    final_model = KMeans(
        n_clusters=FINAL_K,
        n_init=10,
        random_state=42,
    )
    rfm["segment_id"] = final_model.fit_predict(scaled)

    profile = (
        rfm.groupby("segment_id")
        .agg(
            size=("customer_id", "count"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
        )
        .round(2)
        .reset_index()
    )

    # Assign business names based on the measured profiles.
    premium_id = int(
        profile.loc[profile["mean_monetary"].idxmax(), "segment_id"]
    )

    remaining = profile.loc[
        ~profile["segment_id"].eq(premium_id)
    ].copy()

    loyal_id = int(
        remaining.sort_values(
            ["mean_frequency", "mean_recency"],
            ascending=[False, True],
        ).iloc[0]["segment_id"]
    )

    remaining = remaining.loc[
        ~remaining["segment_id"].eq(loyal_id)
    ].copy()

    at_risk_id = int(
        remaining.loc[
            remaining["mean_recency"].idxmax(), "segment_id"
        ]
    )

    name_map = {
        premium_id: "Premium",
        loyal_id: "Loyal Regulars",
        at_risk_id: "At-Risk Occasional",
    }

    for segment_id in profile["segment_id"]:
        segment_id = int(segment_id)
        if segment_id not in name_map:
            name_map[segment_id] = "Developing Customers"

    profile["segment_name"] = profile["segment_id"].map(name_map)
    profile["is_premium"] = profile["segment_id"].eq(premium_id)

    rfm["segment_name"] = rfm["segment_id"].map(name_map)
    rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

    segment_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]

    rfm[segment_columns].to_csv(
        OUTPUTS / "customer_segments.csv", index=False
    )
    profile.to_csv(OUTPUTS / "segment_profile.csv", index=False)

    print("\nSEGMENT PROFILES")
    print(profile.to_string(index=False))
    print("\nPremium segment ID:", premium_id)
    print("Cluster IDs are arbitrary labels, not an ordered ranking.")

    print("\nRECOMMENDED CAMPAIGN ACTIONS")
    actions = {
        "Premium": "Offer VIP benefits and personalised cross-sell campaigns.",
        "Loyal Regulars": "Reward repeat purchases with loyalty benefits.",
        "At-Risk Occasional": "Send a targeted win-back offer and reminder.",
        "Developing Customers": "Encourage a second purchase with relevant recommendations.",
    }
    for name in profile["segment_name"]:
        print(f"{name}: {actions[name]}")

    # PCA on the scaled RFM matrix.
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    ratios = pca.explained_variance_ratio_

    print("\nPCA EXPLAINED VARIANCE RATIO")
    print([round(float(x), 4) for x in ratios])
    print("Total explained variance:", round(float(ratios.sum()), 4))

    # Save the two-dimensional PCA projection for plotting later.
    projection = pd.DataFrame({
        "customer_id": rfm["customer_id"],
        "pca_1": components[:, 0],
        "pca_2": components[:, 1],
        "segment_id": rfm["segment_id"],
        "segment_name": rfm["segment_name"],
    })
    projection.to_csv(OUTPUTS / "customer_pca.csv", index=False)

    print("\nK=8 has lower inertia than K=2 because adding clusters cannot increase")
    print("the minimum within-cluster sum of squares. That alone does not select K:")
    print("larger K values can over-fragment customers, so silhouette and business")
    print("interpretability must also be considered.")

    print("\nCreated:")
    print("outputs/customer_rfm.csv")
    print("outputs/kmeans_metrics.csv")
    print("outputs/segment_profile.csv")
    print("outputs/customer_segments.csv")
    print("outputs/customer_pca.csv")


if __name__ == "__main__":
    main()
