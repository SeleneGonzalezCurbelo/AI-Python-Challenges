# 01 - Territorial Clustering: K-Means and Stability Analysis

## Context
This project analyzes a synthetic dataset of 600 geographical zones to identify groups with similar territorial characteristics. The resulting segments can support policy decisions, for example by highlighting areas with limited access to services or weak public transport coverage.

Because clustering always produces a partition, the number of groups must be justified and the stability of the solution must be checked before interpreting it.

## Mission
Select the number of clusters `k` between 2 and 8 using the silhouette score, fit the final K-Means model, profile the clusters in terms of the original variables, and verify that the partition is stable under bootstrap resampling.

## Learning Objectives
- Handle skewed variables with a `log1p` transformation.
- Scale features with `RobustScaler` so that outliers do not dominate distances.
- Train K-Means for several values of `k` and select the best one with the silhouette score.
- Measure cluster stability with bootstrap resampling and the Adjusted Rand Index (ARI).
- Interpret clusters using the mean of the original variables.
- Project the result to 2D with PCA for visualization.

## Data
Each of the 600 synthetic zones has six features:

| Variable | Meaning |
| :--- | :--- |
| `population` | Inhabitants of the zone |
| `income` | Average annual income (€) |
| `dist_health` | Distance to the nearest health center (km) |
| `dist_education` | Distance to the nearest education center (km) |
| `percentage_elderly` | Population over 65 years old (%) |
| `transport` | Public transport coverage index (0–100) |

## Theoretical Foundation
A log transformation reduces right skew before scaling:

\[
x' = \log(1 + x)
\]

`RobustScaler` centers by the median and divides by the interquartile range, which is less sensitive to extreme values than standardization.

K-Means minimizes the within-cluster sum of squares:

\[
\sum_{k=1}^{K}\sum_{x \in C_k} \|x - \mu_k\|^2
\]

The silhouette score compares intra-cluster distance `a(i)` with the nearest-cluster distance `b(i)`:

\[
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]

Values near 1 indicate well-separated clusters; values near 0 indicate overlapping clusters. The Adjusted Rand Index (ARI) compares two partitions and equals 1 when they are identical and 0 for random agreement.

## Experiment Flow
1. **Preprocessing**: `log1p` applied to skewed variables.
2. **Scaling**: normalization with `RobustScaler`.
3. **Clustering**: K-Means for `k ∈ [2, 8]`.
4. **Evaluation**: selection of the best `k` by silhouette score.
5. **Stability Analysis**: 30 bootstrap resamples, each with its own preprocessing fit, compared against the reference partition with ARI.
6. **Interpretation**: mean of every original variable per cluster.
7. **Visualization**: PCA projection with two components.

## Results
Silhouette scores by number of clusters:

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Silhouette | **0.3862** | 0.2883 | 0.2312 | 0.2346 | 0.2124 | 0.2113 | 0.1987 |

The best score is obtained for `k = 2`.

Bootstrap stability over 30 resamples:

| Metric | Value |
| :--- | ---: |
| Mean ARI | 0.9463 |
| Standard deviation | 0.0436 |
| 5th percentile | 0.8657 |
| 95th percentile | 0.9903 |

Cluster profiles (means of original variables):

| Cluster | Income (€) | Elderly (%) | Transport index |
| :--- | ---: | ---: | ---: |
| 0 | 19,760 | 29.6 | 35.7 |
| 1 | 28,382 | 16.6 | 79.2 |

Cluster 0 groups older, lower-income zones with poor transport coverage, while cluster 1 groups younger, wealthier zones with high coverage. The PCA scatter plot is saved to `results/clusters_plot.png`.

## Requirements
Dependencies are listed in the root `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

## Usage
The complete implementation is in `solution.py`. Run it with:

```bash
python solution.py
```

The script prints the silhouette table, the bootstrap statistics and the cluster profile, then saves the PCA plot. It also runs automatic sanity checks on shapes and values.

## Conclusions and Limitations
- The silhouette score clearly favors two clusters, even though the generator embeds three latent profiles; the two-cluster solution merges two of them. The result should be read as a descriptive partition, not as the recovery of the generating process.
- The high mean ARI (0.9463) shows that the selected partition is stable under resampling. The preprocessing pipeline is refitted inside each bootstrap iteration to avoid information leakage.
- There are no external labels, so cluster quality can only be assessed with internal metrics and interpretability.
- PCA is used only for display; distances are computed in the full six-dimensional scaled space.
