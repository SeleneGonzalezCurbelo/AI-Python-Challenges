import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.utils import resample

SEED = 42
rng = np.random.default_rng(SEED)

# Generation of 600 synthetic zones
n = 600
profile = rng.choice(3, n, p=[0.4, 0.35, 0.25])

income_base = [29000, 19000, 23000]
health_base = [1.5, 7.0, 4.0]
education_base = [1.0, 6.0, 3.0]
elderly_base = [16, 32, 23]
transport_base = [85, 25, 55]

df = pd.DataFrame({
    "population": rng.lognormal(7.2, 0.65, n).astype(int),
    "income": np.array(income_base)[profile]
             + rng.normal(0, 3500, n),
    "dist_health": np.maximum(
        0.1, np.array(health_base)[profile]
        + rng.normal(0, 1.3, n)),
    "dist_education": np.maximum(
        0.1, np.array(education_base)[profile]
        + rng.normal(0, 1.2, n)),
    "percentage_elderly": np.clip(
        np.array(elderly_base)[profile]
        + rng.normal(0, 5, n), 0, 100),
    "transport": np.clip(
        np.array(transport_base)[profile]
        + rng.normal(0, 12, n), 0, 100)
})

# Save a copy for final interpretation (raw data)
df_raw = df.copy()

# Applying log1p to skewed variables
df_transformed = df.copy()
df_transformed["population"] = np.log1p(df_transformed["population"])
df_transformed["dist_health"] = np.log1p(df_transformed["dist_health"])
df_transformed["dist_education"] = np.log1p(df_transformed["dist_education"])

# Scaling variables with RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_transformed)

# Training KMeans for k between 2 and 8
sil_scores = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append({"k": k, "silhouette": score})

results = pd.DataFrame(sil_scores)
print("\nSilhouette Results:")
print(results)

# Selecting the best k, training KMeans, and saving labels
best_k = results.loc[results["silhouette"].idxmax(), "k"]
print(f"\nBest k selected: {best_k}")

final_kmeans = KMeans(n_clusters=int(best_k), random_state=SEED, n_init=10)
df_raw["cluster"] = final_kmeans.fit_predict(X_scaled)
reference_labels = df_raw["cluster"]

# --- Bootstrap Stability Analysis ---
print("\nRunning Bootstrap Stability Analysis...")
ari_scores = []
n_bootstraps = 30
for i in range(n_bootstraps):
    # 1. Generate bootstrap sample
    bootstrap_sample = resample(df, n_samples=n, random_state=i)
    
    # 2. Preprocess (log1p and RobustScaler)
    sample_transformed = bootstrap_sample.copy()
    sample_transformed["population"] = np.log1p(sample_transformed["population"])
    sample_transformed["dist_health"] = np.log1p(sample_transformed["dist_health"])
    sample_transformed["dist_education"] = np.log1p(sample_transformed["dist_education"])
    
    scaler_boot = RobustScaler()
    X_scaled_boot = scaler_boot.fit_transform(sample_transformed)
    
    # Train KMeans on bootstrap sample
    kmeans_boot = KMeans(n_clusters=int(best_k), random_state=i, n_init=10)
    kmeans_boot.fit(X_scaled_boot)
    
    # 3. Predict on full original dataset (applying bootstrap-fit preprocessing)
    full_transformed = df.copy()
    full_transformed["population"] = np.log1p(full_transformed["population"])
    full_transformed["dist_health"] = np.log1p(full_transformed["dist_health"])
    full_transformed["dist_education"] = np.log1p(full_transformed["dist_education"])
    
    X_scaled_full = scaler_boot.transform(full_transformed)
    boot_labels = kmeans_boot.predict(X_scaled_full)
    
    # 4. Compare with Adjusted Rand Index
    ari = adjusted_rand_score(reference_labels, boot_labels)
    ari_scores.append(ari)

# 5. Calculate stats
print(f"\nBootstrap Analysis (n={n_bootstraps}):")
print(f"Mean ARI: {np.mean(ari_scores):.4f}")
print(f"Std ARI: {np.std(ari_scores):.4f}")
print(f"5th Percentile: {np.percentile(ari_scores, 5):.4f}")
print(f"95th Percentile: {np.percentile(ari_scores, 95):.4f}")

# Getting the mean of each ORIGINAL variable per cluster
print("\nMean of original variables per cluster:")
print(df_raw.groupby("cluster").mean())

# PCA (2 components) and scatter plot
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

# Save 'results' within the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df_raw["cluster"], cmap="viridis", alpha=0.6)
plt.title("Cluster Visualization with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.savefig(os.path.join(results_dir, "clusters_plot.png"))
print(f"\nCluster plot saved in '{os.path.join(results_dir, 'clusters_plot.png')}'.")

# Automatic checks
df = df_raw
assert len(df) == 600
assert not df.isna().any().any()
assert X_scaled.shape == (600, 6)
assert np.isfinite(X_scaled).all()
assert 2 <= df["cluster"].nunique() <= 8
assert len(results) == 7
assert results["silhouette"].between(-1, 1).all()

print("\nAll automatic checks passed successfully.")

