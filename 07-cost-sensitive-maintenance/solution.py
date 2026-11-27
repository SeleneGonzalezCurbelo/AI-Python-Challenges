import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
)
from xgboost import XGBClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COST_FALSE_NEGATIVE = 5000
COST_FALSE_POSITIVE = 200
THEORETICAL_THRESHOLD = COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE)


def make_dataset():
    X, y = make_classification(
        n_samples=10000,
        n_features=12,
        n_informative=7,
        n_redundant=3,
        n_repeated=0,
        n_classes=2,
        weights=[0.95, 0.05],
        flip_y=0.01,
        random_state=SEED,
    )
    return X, y


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def economic_cost(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    false_negatives = int(np.sum((y_true == 1) & (y_pred == 0)))
    false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
    return COST_FALSE_NEGATIVE * false_negatives + COST_FALSE_POSITIVE * false_positives


# Backwards-compatible alias.
business_cost = economic_cost


def build_models(y_train):
    neg = int(np.sum(y_train == 0))
    pos = int(np.sum(y_train == 1))
    scale_pos_weight = neg / pos

    models = {
        "Logistic Regression": Pipeline(
            memory=None,
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=SEED,
                    ),
                ),
            ],
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            eval_metric="logloss",
            tree_method="hist",
        ),
    }
    return models


def evaluate_probabilities(y_true, probabilities, threshold=0.5):
    y_pred = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "brier": brier_score_loss(y_true, probabilities),
        "cost": business_cost(y_true, y_pred),
        "false_negatives": int(np.sum((np.asarray(y_true) == 1) & (y_pred == 0))),
        "false_positives": int(np.sum((np.asarray(y_true) == 0) & (y_pred == 1))),
    }


def tune_threshold(y_true, probabilities):
    thresholds = np.linspace(0.0, 1.0, 1001)
    costs = [business_cost(y_true, (probabilities >= t).astype(int)) for t in thresholds]
    best_index = int(np.argmin(costs))
    return float(thresholds[best_index]), int(costs[best_index])


def economic_benefit(probabilities):
    probabilities = np.asarray(probabilities)
    return probabilities * COST_FALSE_NEGATIVE - (1.0 - probabilities) * COST_FALSE_POSITIVE


def evaluate_policies(y_true, probabilities, threshold, capacity=100):
    y_true = np.asarray(y_true)
    benefits = economic_benefit(probabilities)
    predictions = {
        "A_fixed_0.5": (probabilities >= 0.5).astype(int),
        "B_cost_optimized": (probabilities >= threshold).astype(int),
    }

    capacity_predictions = np.zeros_like(y_true, dtype=int)
    eligible = np.where(benefits > 0)[0]
    selected = eligible[np.argsort(benefits[eligible])[::-1][:capacity]]
    capacity_predictions[selected] = 1
    predictions["C_economic_capacity"] = capacity_predictions

    rows = []
    for name, pred in predictions.items():
        rows.append(
            {
                "policy": name,
                "cost": economic_cost(y_true, pred),
                "failures_detected": int(np.sum((y_true == 1) & (pred == 1))),
                "interventions": int(np.sum(pred)),
                "expected_benefit": float(np.sum(benefits[pred == 1])),
            }
        )
    return pd.DataFrame(rows)


def main():
    X, y = make_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    models = build_models(y_train)

    # Cross-validation on train for model selection.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for name, model in models.items():
        ap_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="average_precision"
        )
        f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        recall_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="recall"
        )
        cv_rows.append(
            {
                "model": name,
                "cv_pr_auc_mean": float(np.mean(ap_scores)),
                "cv_pr_auc_std": float(np.std(ap_scores)),
                "cv_f1_mean": float(np.mean(f1_scores)),
                "cv_recall_mean": float(np.mean(recall_scores)),
            }
        )
    cv_df = pd.DataFrame(cv_rows).sort_values("cv_pr_auc_mean", ascending=False)
    best_model_name = cv_df.iloc[0]["model"]

    # Fit all models on train and evaluate on validation.
    fitted_models = {}
    validation_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        val_prob = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_probabilities(y_val, val_prob, threshold=0.5)
        validation_rows.append({"model": name, **metrics})
    validation_df = pd.DataFrame(validation_rows)

    # Calibrate the selected model.
    raw_model = fitted_models[best_model_name]
    calibrated_model = CalibratedClassifierCV(raw_model, method="sigmoid", cv=5)
    calibrated_model.fit(X_train, y_train)

    val_prob_calibrated = calibrated_model.predict_proba(X_val)[:, 1]
    threshold, val_cost = tune_threshold(y_val, val_prob_calibrated)
    theoretical_val_cost = business_cost(
        y_val, (val_prob_calibrated >= THEORETICAL_THRESHOLD).astype(int)
    )

    # Final evaluation on test.
    test_prob_raw = raw_model.predict_proba(X_test)[:, 1]
    test_prob = calibrated_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    test_pred_theoretical = (test_prob >= THEORETICAL_THRESHOLD).astype(int)
    test_metrics = evaluate_probabilities(y_test, test_prob, threshold=threshold)
    _ = evaluate_probabilities(y_test, test_prob_raw, threshold=0.5)
    test_cost_theoretical = business_cost(y_test, test_pred_theoretical)

    policy_df = evaluate_policies(
        y_test, test_prob, threshold, capacity=100
    )
    policy_df.to_csv(os.path.join(RESULTS_DIR, "policy_comparison.csv"), index=False)

    # Baselines.
    baseline_all_negative_cost = business_cost(y_test, np.zeros_like(y_test))
    baseline_threshold_05_cost = business_cost(
        y_test, (test_prob >= 0.5).astype(int)
    )

    # Save metrics.
    pd.DataFrame([{"model": best_model_name, **test_metrics}]).to_csv(
        os.path.join(RESULTS_DIR, "metrics.csv"), index=False
    )

    costs_df = pd.DataFrame(
        [
            {"strategy": "all_negative", "cost": baseline_all_negative_cost},
            {"strategy": "threshold_0.5", "cost": baseline_threshold_05_cost},
            {
                "strategy": f"theoretical_threshold_{THEORETICAL_THRESHOLD:.5f}",
                "cost": test_cost_theoretical,
            },
            {"strategy": f"tuned_threshold_{threshold:.4f}", "cost": test_metrics["cost"]},
        ]
    )
    costs_df.to_csv(os.path.join(RESULTS_DIR, "costs.csv"), index=False)

    thresholds_df = pd.DataFrame(
        [
            {
                "selected_model": best_model_name,
                "threshold": threshold,
                "validation_cost": val_cost,
                "theoretical_threshold": THEORETICAL_THRESHOLD,
                "theoretical_validation_cost": theoretical_val_cost,
            }
        ]
    )
    thresholds_df.to_csv(os.path.join(RESULTS_DIR, "thresholds.csv"), index=False)

    calibration_comparison_df = pd.DataFrame(
        [
            {
                "model": best_model_name,
                "variant": "original",
                "brier": brier_score_loss(y_test, test_prob_raw),
                "pr_auc": average_precision_score(y_test, test_prob_raw),
            },
            {
                "model": best_model_name,
                "variant": "sigmoid_calibrated",
                "brier": brier_score_loss(y_test, test_prob),
                "pr_auc": average_precision_score(y_test, test_prob),
            },
        ]
    )
    calibration_comparison_df.to_csv(
        os.path.join(RESULTS_DIR, "calibration_comparison.csv"), index=False
    )

    cv_df.to_csv(os.path.join(RESULTS_DIR, "cv_results.csv"), index=False)
    validation_df.to_csv(os.path.join(RESULTS_DIR, "validation_results.csv"), index=False)

    # Confusion matrix.
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{best_model_name} @ threshold {threshold:.3f}")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No failure", "Failure"])
    ax.set_yticklabels(["No failure", "Failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # Precision-recall curve.
    precision, recall, _ = precision_recall_curve(y_test, test_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {test_metrics['pr_auc']:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pr_curve.png"), dpi=120)
    plt.close(fig)

    # Cost vs. threshold curve (validation).
    cost_thresholds = np.linspace(0.0, 1.0, 1001)
    cost_values = [
        business_cost(y_val, (val_prob_calibrated >= t).astype(int))
        for t in cost_thresholds
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(cost_thresholds, cost_values, label="Validation cost")
    ax.axvline(
        THEORETICAL_THRESHOLD,
        color="orange",
        linestyle="--",
        label=f"Theoretical = {THEORETICAL_THRESHOLD:.4f}",
    )
    ax.axvline(
        threshold, color="red", linestyle="--", label=f"Empirical = {threshold:.4f}"
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Business cost (€)")
    ax.set_title("Cost vs. threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cost_vs_threshold.png"), dpi=120)
    plt.close(fig)

    # Calibration curve.
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, test_prob_raw, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, test_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred_raw, prob_true_raw, marker="o", label="Original")
    ax.plot(prob_pred_cal, prob_true_cal, marker="o", label="Sigmoid calibrated")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (test)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "calibration.png"), dpi=120)
    plt.close(fig)

    # Report.
    print("=== Cross-validation (train) ===")
    print(cv_df.to_string(index=False))
    print("\n=== Validation (threshold = 0.5) ===")
    print(validation_df.to_string(index=False))
    print(f"\nSelected model: {best_model_name}")
    print(f"Tuned threshold: {threshold:.4f} (validation cost: {val_cost})")
    print("\n=== Test metrics ===")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")
    print("\n=== Cost comparison (test) ===")
    print(costs_df.to_string(index=False))
    print(f"\nConfusion matrix:\n{cm}")

    print("\\n=== Theoretical vs. empirical threshold ===")
    print(f"Theoretical threshold: {THEORETICAL_THRESHOLD:.5f}")
    print(f"Empirical threshold:   {threshold:.5f}")
    print(f"Theoretical test cost: {test_cost_theoretical}")
    print(f"Empirical test cost:   {test_metrics['cost']}")
    print("\n=== Policy comparison (capacity = 100) ===")
    print(policy_df.to_string(index=False))

    print("\\n=== Why Average Precision for model selection ===")
    print(
        "Average Precision evaluates the precision-recall trade-off across all "
        "thresholds, is robust to class imbalance, and measures the quality of "
        "the score ranking that a custom decision threshold will later use. "
        "Accuracy can be misleadingly high by predicting the majority class."
    )

    # Expose variables for automatic tests.
    globals()["X_train"] = X_train
    globals()["X_val"] = X_val
    globals()["X_test"] = X_test
    globals()["y_train"] = y_train
    globals()["y_val"] = y_val
    globals()["y_test"] = y_test
    globals()["economic_cost"] = economic_cost

    # Sanity assertions.
    assert len(X_train) == 6000
    assert len(X_val) == 2000
    assert len(X_test) == 2000
    assert best_model_name in models
    assert 0.0 <= threshold <= 1.0
    assert test_metrics["cost"] <= baseline_all_negative_cost


if __name__ == "__main__":
    main()
