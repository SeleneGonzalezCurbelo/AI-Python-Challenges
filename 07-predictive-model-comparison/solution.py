import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
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
COST_FP = 200
COST_FN = 5000
THEORETICAL_THRESHOLD = COST_FP / (COST_FP + COST_FN)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)

# Data generation.
X, y = make_classification(
    n_samples=10000,
    n_features=12,
    n_informative=7,
    n_redundant=3,
    n_classes=2,
    weights=[0.95, 0.05],
    class_sep=1.1,
    flip_y=0.01,
    random_state=SEED,
)

# 60% train, 20% validation, 20% test.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("Train prevalence:", y_train.mean())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Model definitions with the requested initial parameters.
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

# Cross-validation comparison.
scoring = {
    "average_precision": "average_precision",
    "f1": "f1",
    "recall": "recall",
}
cv_results = []
for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    cv_results.append(
        {
            "model": name,
            "cv_pr_auc_mean": float(np.mean(scores["test_average_precision"])),
            "cv_pr_auc_std": float(np.std(scores["test_average_precision"])),
            "cv_f1_mean": float(np.mean(scores["test_f1"])),
            "cv_recall_mean": float(np.mean(scores["test_recall"])),
        }
    )
cv_df = pd.DataFrame(cv_results).sort_values("cv_pr_auc_mean", ascending=False)
best_model_name = cv_df.iloc[0]["model"]

# Train all models and evaluate on validation.
fitted_models = {}
validation_rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model
    val_prob = model.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    validation_rows.append(
        {
            "model": name,
            "accuracy": accuracy_score(y_val, val_pred),
            "precision": precision_score(y_val, val_pred, zero_division=0),
            "recall": recall_score(y_val, val_pred, zero_division=0),
            "f1": f1_score(y_val, val_pred, zero_division=0),
            "pr_auc": average_precision_score(y_val, val_prob),
            "roc_auc": roc_auc_score(y_val, val_prob),
            "brier": brier_score_loss(y_val, val_prob),
        }
    )
validation_df = pd.DataFrame(validation_rows)

# Calibrate the selected model on training data only.
raw_model = fitted_models[best_model_name]
calibrated_model = CalibratedClassifierCV(raw_model, method="sigmoid", cv=5)
calibrated_model.fit(X_train, y_train)

val_prob_raw = raw_model.predict_proba(X_val)[:, 1]
val_prob_calibrated = calibrated_model.predict_proba(X_val)[:, 1]


def economic_cost(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    false_negatives = int(np.sum((y_true == 1) & (y_pred == 0)))
    false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
    return COST_FN * false_negatives + COST_FP * false_positives


def optimize_threshold(y_true, probabilities):
    thresholds = np.linspace(0.0, 1.0, 1001)
    costs = [
        economic_cost(y_true, (probabilities >= threshold).astype(int))
        for threshold in thresholds
    ]
    best_index = int(np.argmin(costs))
    return float(thresholds[best_index]), int(costs[best_index])


def economic_benefit(probabilities):
    probabilities = np.asarray(probabilities)
    return probabilities * COST_FN - (1.0 - probabilities) * COST_FP


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


empirical_threshold, validation_cost = optimize_threshold(y_val, val_prob_calibrated)
theoretical_validation_cost = economic_cost(
    y_val, (val_prob_calibrated >= THEORETICAL_THRESHOLD).astype(int)
)

# Test evaluation.
test_prob_raw = raw_model.predict_proba(X_test)[:, 1]
test_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
test_pred_calibrated = (test_prob_calibrated >= empirical_threshold).astype(int)
test_pred_theoretical = (test_prob_calibrated >= THEORETICAL_THRESHOLD).astype(int)

baseline_pred = np.zeros_like(y_test)
baseline_cost = economic_cost(y_test, baseline_pred)

cost_05 = economic_cost(y_test, (test_prob_calibrated >= 0.5).astype(int))
cost_calibrated_05 = economic_cost(y_test, (test_prob_calibrated >= 0.5).astype(int))
cost_empirical = economic_cost(y_test, test_pred_calibrated)
cost_theoretical = economic_cost(y_test, test_pred_theoretical)

policy_df = evaluate_policies(
    y_test, test_prob_calibrated, empirical_threshold, capacity=100
)
policy_df.to_csv(RESULTS_DIR / "policy_comparison.csv", index=False)

# Save comparison results.
comparison_df = pd.DataFrame(
    [
        {"strategy": "baseline_always_normal", "cost": baseline_cost},
        {"strategy": "uncalibrated_threshold_0.5", "cost": economic_cost(y_test, (test_prob_raw >= 0.5).astype(int))},
        {"strategy": "calibrated_threshold_0.5", "cost": cost_calibrated_05},
        {"strategy": f"calibrated_threshold_{empirical_threshold:.5f}", "cost": cost_empirical},
        {"strategy": f"calibrated_theoretical_{THEORETICAL_THRESHOLD:.5f}", "cost": cost_theoretical},
    ]
)
comparison_df.to_csv(RESULTS_DIR / "cost_comparison.csv", index=False)

thresholds_df = pd.DataFrame(
    [
        {
            "selected_model": best_model_name,
            "empirical_threshold": empirical_threshold,
            "theoretical_threshold": THEORETICAL_THRESHOLD,
            "validation_cost_empirical": validation_cost,
            "validation_cost_theoretical": theoretical_validation_cost,
        }
    ]
)
thresholds_df.to_csv(RESULTS_DIR / "thresholds.csv", index=False)

cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
validation_df.to_csv(RESULTS_DIR / "validation_results.csv", index=False)

calibration_comparison_df = pd.DataFrame(
    [
        {
            "variant": "original",
            "brier": brier_score_loss(y_test, test_prob_raw),
            "pr_auc": average_precision_score(y_test, test_prob_raw),
        },
        {
            "variant": "sigmoid_calibrated",
            "brier": brier_score_loss(y_test, test_prob_calibrated),
            "pr_auc": average_precision_score(y_test, test_prob_calibrated),
        },
    ]
)
calibration_comparison_df.to_csv(RESULTS_DIR / "calibration_comparison.csv", index=False)

# Visualizations.
precision, recall, _ = precision_recall_curve(y_test, test_prob_calibrated)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall, precision, label=f"AP = {average_precision_score(y_test, test_prob_calibrated):.4f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curve (test)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "pr_curve.png", dpi=120)
plt.close(fig)

prob_true_raw, prob_pred_raw = calibration_curve(y_test, test_prob_raw, n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(y_test, test_prob_calibrated, n_bins=10)
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
fig.savefig(RESULTS_DIR / "calibration.png", dpi=120)
plt.close(fig)

cost_thresholds = np.linspace(0.0, 1.0, 1001)
cost_values = [
    economic_cost(y_val, (val_prob_calibrated >= threshold).astype(int))
    for threshold in cost_thresholds
]
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(cost_thresholds, cost_values, label="Validation cost")
ax.axvline(THEORETICAL_THRESHOLD, color="orange", linestyle="--", label=f"Theoretical = {THEORETICAL_THRESHOLD:.4f}")
ax.axvline(empirical_threshold, color="red", linestyle="--", label=f"Empirical = {empirical_threshold:.4f}")
ax.set_xlabel("Threshold")
ax.set_ylabel("Business cost")
ax.set_title("Cost vs. threshold")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "cost_vs_threshold.png", dpi=120)
plt.close(fig)

cm = confusion_matrix(y_test, test_pred_calibrated, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_title(f"{best_model_name} @ threshold {empirical_threshold:.3f}")
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
fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=120)
plt.close(fig)

print("Cross-validation results:")
print(cv_df.to_string(index=False))
print("\nValidation results:")
print(validation_df.to_string(index=False))
print(f"\nSelected model: {best_model_name}")
print(f"Empirical threshold: {empirical_threshold:.5f}")
print(f"Theoretical threshold: {THEORETICAL_THRESHOLD:.5f}")
print(f"Baseline cost: {baseline_cost}")
print(f"Uncalibrated 0.5 cost: {economic_cost(y_test, (test_prob_raw >= 0.5).astype(int))}")
print(f"Calibrated 0.5 cost: {cost_calibrated_05}")
print(f"Calibrated empirical cost: {cost_empirical}")
print(f"Calibrated theoretical cost: {cost_theoretical}")
print("\nPolicy comparison (capacity = 100):")
print(policy_df.to_string(index=False))
print("\nConfusion matrix:")
print(cm)

# Expose variables for automatic tests.
globals()["X_train"] = X_train
globals()["X_val"] = X_val
globals()["X_test"] = X_test
globals()["y_train"] = y_train
globals()["y_val"] = y_val
globals()["y_test"] = y_test
globals()["economic_cost"] = economic_cost

# Sanity checks.
assert X_train.shape == (6000, 12)
assert X_val.shape == (2000, 12)
assert X_test.shape == (2000, 12)
assert 0.03 < y_train.mean() < 0.08
assert economic_cost([0, 1, 0, 1], [0, 1, 1, 0]) == 5200
assert economic_cost([0, 1], [0, 1]) == 0
assert economic_cost([0, 0], [1, 1]) == 400
assert economic_cost([1, 1], [0, 0]) == 10000