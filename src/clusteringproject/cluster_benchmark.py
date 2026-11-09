import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import HDBSCAN, KMeans

from sklearn.metrics import (
     adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
BENCHMARK_DIR = PROCESSED_DIR / "benchmark"

K_GRID = [10, 12, 15, 20, 25, 30, 40, 50, 60]
PRIMARY_K = 30
MACRO_K = 12


def compute_wcss(X: np.ndarray, labels: np.ndarray) -> float:
    """Computes Within-Cluster Sum of Squares (Inertia W) for any partition."""
    wcss = 0.0
    for k in np.unique(labels):
        if k == -1:  # Ignore HDBSCAN noise points in WCSS
            continue
        cluster_pts = X[labels == k]
        if len(cluster_pts) > 0:
            centroid = cluster_pts.mean(axis=0)
            wcss += float(np.sum((cluster_pts - centroid) ** 2))
    return wcss


def compute_cah_barycenters(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Computes the K cluster barycenters (mean vectors) from CAH labels to seed K-Means."""
    unique_labels = np.unique(labels)
    barycenters = np.zeros((len(unique_labels), X.shape[1]), dtype=np.float32)
    for idx, lbl in enumerate(unique_labels):
        barycenters[idx] = X[labels == lbl].mean(axis=0)
    return barycenters


def evaluate_partition(
    X: np.ndarray,
    labels: np.ndarray,
    algo_name: str,
    k_target: int,
    elapsed_sec: float,
) -> dict:
    """Calculates internal validation metrics for a clustering partition."""
    # Filter out noise (-1) if present (for HDBSCAN fair comparison on core clusters)
    valid_mask = labels != -1
    n_noise = int(np.sum(~valid_mask))
    noise_ratio = n_noise / len(labels)

    unique_valid = np.unique(labels[valid_mask])
    n_clusters_found = len(unique_valid)

    if n_clusters_found < 2:
        return {
            "algorithm": algo_name,
            "k_target": k_target,
            "k_found": n_clusters_found,
            "wcss_inertia": np.nan,
            "silhouette_euclidean": np.nan,
            "silhouette_cosine": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "max_cluster_share": 1.0,
            "noise_ratio": round(noise_ratio, 4),
            "time_sec": round(elapsed_sec, 3),
        }

    X_eval = X[valid_mask]
    labels_eval = labels[valid_mask]

    wcss = compute_wcss(X_eval, labels_eval)
    sil_euc = float(silhouette_score(X_eval, labels_eval, metric="euclidean"))
    sil_cos = float(silhouette_score(X_eval, labels_eval, metric="cosine"))
    dbi = float(davies_bouldin_score(X_eval, labels_eval))
    chi = float(calinski_harabasz_score(X_eval, labels_eval))

    # Track the largest cluster share to quantify Single Linkage chaining effect
    _, counts = np.unique(labels_eval, return_counts=True)
    max_share = float(counts.max() / len(labels_eval))

    return {
        "algorithm": algo_name,
        "k_target": k_target,
        "k_found": n_clusters_found,
        "wcss_inertia": round(wcss, 2),
        "silhouette_euclidean": round(sil_euc, 4),
        "silhouette_cosine": round(sil_cos, 4),
        "davies_bouldin": round(dbi, 4),
        "calinski_harabasz": round(chi, 2),
        "max_cluster_share": round(max_share, 4),
        "noise_ratio": round(noise_ratio, 4),
        "time_sec": round(elapsed_sec, 3),
    }


def run_benchmark_pipeline() -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    X_path = PROCESSED_DIR / "X_pca50.npy"
    if not X_path.exists():
        raise FileNotFoundError(f"Missing {X_path}. Run reduce_dimensions.py first.")

    X = np.load(X_path)
    df_movies = pd.read_parquet(PROCESSED_DIR / "movies_index.parquet")
    print(f"Loaded X_pca50 of shape {X.shape} for clustering benchmark.")

    metrics_records: list[dict] = []
    saved_labels: dict[str, np.ndarray] = {}

    # -------------------------------------------------------------------------
    # 1. Precompute CAH Linkage Trees (O(N^2) distance matrix computed once)
    # -------------------------------------------------------------------------
    print("\n--- Precomputing CAH Hierarchical Linkage Trees ---")
    t0 = time.perf_counter()
    pdist_euc = pdist(X, metric="euclidean")
    dist_time = time.perf_counter() - t0
    print(f"Computed pairwise Euclidean distance vector in {dist_time:.2f}s")

    linkage_trees: dict[str, tuple[np.ndarray, float]] = {}
    for method in ["ward", "complete", "average", "single"]:
        t_start = time.perf_counter()
        Z = linkage(pdist_euc, method=method)
        elapsed = (time.perf_counter() - t_start) + dist_time
        linkage_trees[method] = (Z, elapsed)
        print(f"  Built CAH ({method}) linkage tree in {elapsed:.2f}s")

    # Save Ward linkage matrix for dendrogram plotting in Phase 5
    np.save(BENCHMARK_DIR / "cah_ward_linkage.npy", linkage_trees["ward"][0])

    # -------------------------------------------------------------------------
    # 2. Grid Benchmark Across K_GRID
    # -------------------------------------------------------------------------
    print("\n--- Running Grid Benchmark Across K ---")
    for k in K_GRID:
        print(f"Evaluating K = {k}...")

        # A. Standard K-Means (k-means++)
        t_km = time.perf_counter()
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km_labels = km.fit_predict(X)
        t_km_elapsed = time.perf_counter() - t_km
        metrics_records.append(
            evaluate_partition(X, km_labels, "KMeans_PlusPlus", k, t_km_elapsed)
        )

        # B. CAH across all 4 linkages (sliced from precomputed trees)
        cah_ward_labels = None
        for method, (Z, tree_time) in linkage_trees.items():
            # fcluster is 1-indexed; shift to 0-indexed [0, k-1]
            cah_lbl = fcluster(Z, t=k, criterion="maxclust") - 1
            metrics_records.append(
                evaluate_partition(X, cah_lbl, f"CAH_{method.capitalize()}", k, tree_time)
            )
            if method == "ward":
                cah_ward_labels = cah_lbl

            # Save primary map slices
            if k == PRIMARY_K:
                saved_labels[f"cah_{method}_{PRIMARY_K}"] = cah_lbl
            if method == "ward" and k == MACRO_K:
                saved_labels[f"cah_ward_macro_{MACRO_K}"] = cah_lbl

        # C. Hybrid CAH (Ward) + K-Means Consolidation
        t_cons = time.perf_counter()
        barycenters = compute_cah_barycenters(X, cah_ward_labels, k)
        km_cons = KMeans(
            n_clusters=k,
            init=barycenters,
            n_init=1,
            max_iter=300,
            random_state=42,
        )
        cons_labels = km_cons.fit_predict(X)
        t_cons_elapsed = (time.perf_counter() - t_cons) + linkage_trees["ward"][1]
        metrics_records.append(
            evaluate_partition(
                X, cons_labels, "Hybrid_CAH_KMeans_Consolidated", k, t_cons_elapsed
            )
        )

        # Save primary K=30 assignments and point-level diagnostics
        if k == PRIMARY_K:
            saved_labels[f"kmeans_{PRIMARY_K}"] = km_labels
            saved_labels[f"consolidated_{PRIMARY_K}"] = cons_labels

            # Distance to assigned K-Means centroid (for outlier pull analysis)
            km_dists = np.linalg.norm(X - km.cluster_centers_[km_labels], axis=1)
            saved_labels["kmeans_centroid_dist"] = km_dists.astype(np.float32)

            # Per-movie Silhouette score for K=30
            saved_labels["kmeans_silhouette"] = silhouette_samples(
                X, km_labels, metric="euclidean"
            ).astype(np.float32)
            saved_labels["consolidated_silhouette"] = silhouette_samples(
                X, cons_labels, metric="euclidean"
            ).astype(np.float32)

    # -------------------------------------------------------------------------
    # 3. HDBSCAN Density Baseline
    # -------------------------------------------------------------------------
    print("\n--- Running HDBSCAN Baseline ---")
    t_hdb = time.perf_counter()
    hdb = HDBSCAN(min_cluster_size=25, min_samples=10, metric="euclidean")
    hdb_labels = hdb.fit_predict(X)
    t_hdb_elapsed = time.perf_counter() - t_hdb

    n_hdb_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    metrics_records.append(
        evaluate_partition(X, hdb_labels, "HDBSCAN_Baseline", n_hdb_clusters, t_hdb_elapsed)
    )
    saved_labels["hdbscan"] = hdb_labels

    # -------------------------------------------------------------------------
    # 4. Compute Cross-Algorithm Agreement (ARI & NMI at K=30)
    # -------------------------------------------------------------------------
    primary_keys = [
        f"kmeans_{PRIMARY_K}",
        f"cah_ward_{PRIMARY_K}",
        f"consolidated_{PRIMARY_K}",
        f"cah_complete_{PRIMARY_K}",
        f"cah_average_{PRIMARY_K}",
        f"cah_single_{PRIMARY_K}",
        "hdbscan",
    ]
    agreement_rows = []
    for i, k1 in enumerate(primary_keys):
        for k2 in primary_keys[i + 1 :]:
            ari = float(adjusted_rand_score(saved_labels[k1], saved_labels[k2]))
            nmi = float(normalized_mutual_info_score(saved_labels[k1], saved_labels[k2]))
            agreement_rows.append(
                {
                    "algo_1": k1,
                    "algo_2": k2,
                    "ARI": round(ari, 4),
                    "NMI": round(nmi, 4),
                }
            )

    # -------------------------------------------------------------------------
    # 5. Persist All Benchmark Outputs
    # -------------------------------------------------------------------------
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(BENCHMARK_DIR / "metrics_grid.csv", index=False)

    df_agreement = pd.DataFrame(agreement_rows)
    df_agreement.to_csv(BENCHMARK_DIR / "algorithm_agreement.csv", index=False)

    for col_name, arr in saved_labels.items():
        df_movies[col_name] = arr

    df_movies.to_parquet(BENCHMARK_DIR / "cluster_assignments.parquet", index=False)

    print("\n=== Phase 4 Benchmark Summary (K = 30 & HDBSCAN) ===")
    summary_view = df_metrics[
        (df_metrics["k_target"] == PRIMARY_K)
        | (df_metrics["algorithm"] == "HDBSCAN_Baseline")
    ][
        [
            "algorithm",
            "k_found",
            "wcss_inertia",
            "silhouette_euclidean",
            "davies_bouldin",
            "max_cluster_share",
            "noise_ratio",
        ]
    ]
    print(summary_view.to_string(index=False))
    print(f"\nSaved all benchmark artifacts to {BENCHMARK_DIR}")


if __name__ == "__main__":
    run_benchmark_pipeline()