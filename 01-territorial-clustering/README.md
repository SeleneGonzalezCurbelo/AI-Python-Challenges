# 01 - Territorial Clustering

This project analyzes a synthetic dataset of 600 geographical zones to identify clusters with similar characteristics and evaluate their stability for policy intervention.

## Data
The available features for each zone are:

| Variable | Meaning |
| :--- | :--- |
| `population` | Inhabitants of the zone |
| `income` | Average annual income (€) |
| `dist_health` | Distance to the nearest health center (km) |
| `dist_education` | Distance to the nearest education center (km) |
| `percentage_elderly` | Population over 65 years old (%) |
| `transport` | Public transport coverage index (0–100) |

## Analysis Pipeline
The processing pipeline consists of the following steps:

1. **Preprocessing:** Application of logarithmic transformation (`log1p`) to skewed variables.
2. **Scaling:** Data normalization using `RobustScaler`.
3. **Clustering:** Implementation of `KMeans` testing different numbers of groups ($k \in [2, 8]$).
4. **Evaluation:** Selection of the optimal number of clusters using `silhouette_score`.
5. **Stability Analysis:** Robustness assessment using Bootstrap resampling (30 iterations) and Adjusted Rand Index (ARI).
6. **Interpretation:** Analysis of average characteristics per cluster.
7. **Visualization:** Dimensionality reduction with `PCA` to visualize the obtained clusters.

## Requirements
To run this project, ensure you have the dependencies listed in the `requirements.txt` file at the root of the project:

```bash
pip install -r ../requirements.txt
```

## Usage
The complete implementation is in `solution.py`. Run it with:

```bash
python solution.py
```
This script performs the full analysis, runs the bootstrap stability check, and generates cluster plots.

## Bootstrap Stability Analysis
To ensure that the identified clusters are robust, the analysis includes a bootstrap resampling approach (30 iterations) combined with the Adjusted Rand Index (ARI). This evaluates how consistently the pipeline identifies the same groups under slight data variations.
