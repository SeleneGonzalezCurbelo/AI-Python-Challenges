# 07 - Predictive Model Comparison

## Context

An industrial company monitors machinery with sensors and wants to anticipate failures. Only a small proportion of observations are failures, and prediction errors have different costs:

- **False negative (5,000 €)**: a real failure is not detected.
- **False positive (200 €)**: an unnecessary intervention is performed.

## Mission

Compare Logistic Regression, Random Forest and XGBoost on an imbalanced synthetic maintenance dataset, calibrate probabilities, tune the decision threshold on validation, and evaluate once on test.

## Learning Objectives

1. Handle imbalanced classification.
2. Use stratified train/validation/test splits.
3. Compare models with cross-validation and Average Precision.
4. Calibrate probabilistic predictions.
5. Tune the decision threshold to minimize business cost.
6. Avoid data leakage across selection, calibration, threshold tuning and final evaluation.

## Experiment Flow

1. **Data Generation**: 10,000 samples, 12 features, ≈5% positives.
2. **Split**: 6,000 train / 2,000 validation / 2,000 test.
3. **Model Selection**: 5-fold stratified cross-validation by Average Precision.
4. **Calibration**: sigmoid calibration of the selected model on training data.
5. **Threshold Tuning**: minimize validation cost over a grid of thresholds.
6. **Test Evaluation**: one final evaluation with the fixed model and threshold.

## Results

Artifacts are saved under `results/`:

- `results/cv_results.csv`
- `results/validation_results.csv`
- `results/cost_comparison.csv`
- `results/thresholds.csv`
- `results/calibration_comparison.csv`
- `results/pr_curve.png`
- `results/calibration.png`
- `results/cost_vs_threshold.png`
- `results/confusion_matrix.png`

## Requirements

```bash
pip install -r ../requirements.txt
```

## Usage

```bash
python solution.py
```

## Conclusions and Limitations

- Accuracy alone is misleading for imbalanced, cost-sensitive problems.
- Average Precision is preferred for model selection when a custom threshold will be used.
- The dataset is synthetic and does not represent a real physical machine.