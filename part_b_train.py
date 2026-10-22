
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
OUTPUTS = BASE / "outputs"
CLEAN = BASE / "data" / "clean"

NUMERIC = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]
CATEGORICAL = ["city", "gender", "favourite_category"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "lapsed"


def make_pipeline(model):
    # Every preprocessing step is fitted within the training fold.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def report_metrics(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Lapsed precision:",
        round(precision_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Lapsed recall:",
        round(recall_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Lapsed F1:",
        round(f1_score(y_test, predictions, zero_division=0), 4),
    )

    return predictions, probabilities


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(OUTPUTS / "lapse_features.csv")
    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        data["customer_id"],
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Explicit check: no missing features reach model training.
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing feature values found before preprocessing.")

    print("TRAINING CLASS COUNTS")
    print(y_train.value_counts().sort_index().to_dict())
    print("\nTEST CLASS COUNTS")
    print(y_test.value_counts().sort_index().to_dict())

    # Baseline: always predict not lapsed.
    baseline = np.zeros(len(y_test), dtype=int)
    print("\nBASELINE — ALWAYS PREDICT NOT LAPSED")
    print("Accuracy:", round(accuracy_score(y_test, baseline), 4))
    print(
        "Lapsed F1:",
        round(f1_score(y_test, baseline, zero_division=0), 4),
    )

    # Logistic Regression grid search.
    lr_pipeline = make_pipeline(LogisticRegression(max_iter=1000))
    lr_grid = GridSearchCV(
        lr_pipeline,
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    print("\nTUNING LOGISTIC REGRESSION...")
    lr_grid.fit(X_train, y_train)

    print("Logistic best parameters:", lr_grid.best_params_)
    print("Logistic best CV F1:", round(lr_grid.best_score_, 4))

    # Random Forest grid search: 3 x 3 x 5 folds = 45 fits.
    rf_pipeline = make_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
    )
    rf_grid = GridSearchCV(
        rf_pipeline,
        param_grid={
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    print("\nTUNING RANDOM FOREST...")
    rf_grid.fit(X_train, y_train)

    print("Random Forest best parameters:", rf_grid.best_params_)
    print("Random Forest best CV F1:", round(rf_grid.best_score_, 4))

    # Evaluate both only after tuning is complete.
    lr_predictions, lr_probabilities = report_metrics(
        "TUNED LOGISTIC REGRESSION — TEST SET",
        lr_grid.best_estimator_,
        X_test,
        y_test,
    )
    rf_predictions, rf_probabilities = report_metrics(
        "TUNED RANDOM FOREST — TEST SET",
        rf_grid.best_estimator_,
        X_test,
        y_test,
    )

    # The assignment's reference and feature-importance task favour RF.
    selected_name = "Random Forest"
    selected_model = rf_grid.best_estimator_
    selected_predictions = rf_predictions
    selected_probabilities = rf_probabilities
    selected_f1 = f1_score(
        y_test, selected_predictions, zero_division=0
    )

    if selected_f1 < 0.55:
        raise ValueError(
            f"Selected Random Forest test F1={selected_f1:.3f}, "
            "below the required 0.55. Inspect the results before proceeding."
        )

    print(f"\nSELECTED MODEL: {selected_name}")
    print("Selected test F1:", round(selected_f1, 4))
    print(
        "Missing a lapsing customer can mean lost future revenue; "
        "wrongly targeting a loyal customer can waste incentives and annoy them. "
        "The selected model provides a balance of detection performance and "
        "interpretable feature importance, but campaign costs should guide the "
        "final operating threshold."
    )

    # Class-balance experiment: oversample the training set only.
    # Keep every original training row and duplicate lapsed rows with replacement.
    train_df = X_train.copy()
    train_df[TARGET] = y_train.to_numpy()

    class_counts = train_df[TARGET].value_counts()
    majority_count = int(class_counts.max())
    minority_label = int(class_counts.idxmin())
    minority_rows = train_df.loc[train_df[TARGET] == minority_label]

    additions_needed = majority_count - len(minority_rows)
    additions = minority_rows.sample(
        n=additions_needed,
        replace=True,
        random_state=42,
    )

    balanced_train = pd.concat(
        [train_df, additions],
        ignore_index=True,
    )

    X_balanced = balanced_train[FEATURES]
    y_balanced = balanced_train[TARGET].astype(int)

    print("\nBALANCED TRAINING COUNTS")
    print(y_balanced.value_counts().sort_index().to_dict())

    balanced_lr = make_pipeline(
        LogisticRegression(max_iter=1000, C=lr_grid.best_params_["model__C"])
    )
    balanced_lr.fit(X_balanced, y_balanced)

    balanced_predictions = balanced_lr.predict(X_test)

    print("\nLOGISTIC REGRESSION — ORIGINAL TEST SET, AFTER OVERSAMPLING")
    print(
        "Precision:",
        round(precision_score(y_test, balanced_predictions, zero_division=0), 4),
    )
    print(
        "Recall:",
        round(recall_score(y_test, balanced_predictions, zero_division=0), 4),
    )
    print(
        "F1:",
        round(f1_score(y_test, balanced_predictions, zero_division=0), 4),
    )
    print(
        "Unbalanced LR precision / recall / F1:",
        round(precision_score(y_test, lr_predictions, zero_division=0), 4),
        round(recall_score(y_test, lr_predictions, zero_division=0), 4),
        round(f1_score(y_test, lr_predictions, zero_division=0), 4),
    )

    # Random Forest feature importance.
    preprocessor = selected_model.named_steps["preprocessor"]
    rf = selected_model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTOP 5 RANDOM FOREST FEATURES")
    print(importance.head(5).to_string(index=False))
    importance.to_csv(
        OUTPUTS / "random_forest_feature_importance.csv", index=False
    )

    # Logistic coefficient signs for three numeric features.
    lr_model = lr_grid.best_estimator_
    lr_names = lr_model.named_steps["preprocessor"].get_feature_names_out()
    lr_coefficients = lr_model.named_steps["model"].coef_[0]
    coefficient_table = pd.DataFrame({
        "feature": lr_names,
        "coefficient": lr_coefficients,
    })

    print("\nLOGISTIC REGRESSION COEFFICIENTS")
    for feature in [
        "numeric__recency",
        "numeric__frequency",
        "numeric__monetary",
    ]:
        row = coefficient_table.loc[
            coefficient_table["feature"].eq(feature)
        ]
        if not row.empty:
            value = float(row["coefficient"].iloc[0])
            print(f"{feature}: {value:.4f} ({'positive' if value > 0 else 'negative' if value < 0 else 'zero'})")

    coefficient_table.to_csv(
        OUTPUTS / "logistic_coefficients.csv", index=False
    )

    # Persist and reload the selected model, then verify exact predictions.
    model_path = OUTPUTS / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)

    reloaded_model = joblib.load(model_path)
    reloaded_predictions = reloaded_model.predict(X_test)

    predictions_match = np.array_equal(
        selected_predictions, reloaded_predictions
    )
    print("\nRELOADED MODEL PREDICTIONS MATCH:", predictions_match)

    if not predictions_match:
        raise AssertionError("Reloaded model predictions differ.")

    # Retention list: test-set customers who belong to the whole-window Premium
    # segment. This list is post-hoc/illustrative, not leakage-free evaluation.
    segments = pd.read_csv(OUTPUTS / "customer_segments.csv")
    premium_ids = set(
        segments.loc[segments["is_premium"].astype(bool), "customer_id"]
    )

    retention = pd.DataFrame({
        "customer_id": ids_test.to_numpy(),
        "lapse_probability": selected_probabilities,
        "actually_lapsed": y_test.to_numpy(),
    })

    retention = retention.loc[
        retention["customer_id"].isin(premium_ids)
    ].sort_values("lapse_probability", ascending=False).head(10)

    retention.to_csv(OUTPUTS / "retention_list.csv", index=False)

    print("\nPREMIUM RETENTION LIST")
    print(retention.to_string(index=False))
    print("Rows in retention list:", len(retention))
    print("Actually lapsed in list:", int(retention["actually_lapsed"].sum()))

    print("\nSaved:")
    print("outputs/lapse_model.joblib")
    print("outputs/retention_list.csv")
    print("outputs/random_forest_feature_importance.csv")
    print("outputs/logistic_coefficients.csv")


if __name__ == "__main__":
    main()
