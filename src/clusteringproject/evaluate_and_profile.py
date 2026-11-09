import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
BENCHMARK_DIR = PROCESSED_DIR / "benchmark"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

PRIMARY_K = 30
MACRO_K = 12


def apply_dark_theme(fig: plt.Figure, axes: list[plt.Axes] | plt.Axes) -> None:
    """Applies a consistent dark aesthetic to Matplotlib figures."""
    fig.patch.set_facecolor("#0f172a")
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in np.ravel(axes):
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.xaxis.label.set_color("#cbd5e1")
        ax.yaxis.label.set_color("#cbd5e1")
        ax.title.set_color("#f8fafc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.grid(True, color="#1e293b", linestyle="--", linewidth=0.6, alpha=0.7)


def plot_internal_metrics_curves(df_metrics: pd.DataFrame, out_dir: Path) -> None:
    """
    Plots 2x2 grid comparing K-Means, CAH linkages, and Hybrid Consolidation across K:
    1. Within-Cluster Inertia (WCSS)
    2. Silhouette Score (Euclidean)
    3. Davies-Bouldin Index (DBI)
    4. Max Cluster Share (Chaining Effect diagnostic)
    """
    df_k = df_metrics[df_metrics["algorithm"] != "HDBSCAN_Baseline"].copy()

    algo_styles = {
        "KMeans_PlusPlus": ("K-Means (k-means++)", "#38bdf8", "o", "-"),
        "Hybrid_CAH_KMeans_Consolidated": ("Hybrid (CAH Ward + K-Means)", "#4ade80", "s", "--"),
        "CAH_Ward": ("CAH (Ward)", "#a855f7", "^", "-"),
        "CAH_Complete": ("CAH (Complete)", "#facc15", "d", "-."),
        "CAH_Average": ("CAH (Average)", "#fb923c", "v", ":"),
        "CAH_Single": ("CAH (Single)", "#f43f5e", "x", ":"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    apply_dark_theme(fig, axes)

    metrics_cfg = [
        (axes[0, 0], "wcss_inertia", "Within-Cluster Inertia (WCSS - Lower is Tighter)", "Inertia (W)"),
        (axes[0, 1], "silhouette_euclidean", "Silhouette Coefficient (Higher is Better)", "Silhouette Score"),
        (axes[1, 0], "davies_bouldin", "Davies-Bouldin Index (Lower is Better)", "DBI"),
        (axes[1, 1], "max_cluster_share", "Largest Cluster Share (Chaining Diagnostic)", "Share of Dataset (0-1)"),
    ]

    for ax, col, title, ylabel in metrics_cfg:
        for algo_key, (label, color, marker, ls) in algo_styles.items():
            sub = df_k[df_k["algorithm"] == algo_key].sort_values("k_target")
            if sub.empty:
                continue
            ax.plot(
                sub["k_target"],
                sub[col],
                label=label,
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=1.8,
                markersize=5,
            )
        ax.axvline(PRIMARY_K, color="#64748b", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Number of Clusters (K)", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)

    leg = axes[0, 0].legend(
        loc="upper right",
        frameon=True,
        facecolor="#1e293b",
        edgecolor="#334155",
        fontsize=8.5,
    )
    for text in leg.get_texts():
        text.set_color("#e2e8f0")

    fig.suptitle(
        "Quantitative Benchmark Across K: K-Means vs. CAH Linkages vs. Consolidated Hybrid",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "02_internal_metrics_curves.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved metrics curves to {out_path}")


def plot_cah_dendrogram_and_jumps(Z_ward: np.ndarray, out_dir: Path) -> None:
    """
    Plots the truncated CAH Ward Dendrogram alongside the merge distance jump curve
    to justify macro (K=12) and micro (K=30) hierarchy cuts.
    """
    fig, (ax_dendro, ax_jump) = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    apply_dark_theme(fig, [ax_dendro, ax_jump])

    # Left: Truncated Ward Dendrogram (top 40 leaves)
    dendrogram(
        Z_ward,
        truncate_mode="lastp",
        p=40,
        ax=ax_dendro,
        leaf_rotation=90.0,
        leaf_font_size=7.5,
        show_contracted=True,
        above_threshold_color="#94a3b8",
    )
    # Cut heights for K=12 and K=30
    cut_k12 = 0.5 * (Z_ward[-MACRO_K, 2] + Z_ward[-(MACRO_K - 1), 2])
    cut_k30 = 0.5 * (Z_ward[-PRIMARY_K, 2] + Z_ward[-(PRIMARY_K - 1), 2])

    ax_dendro.axhline(cut_k12, color="#f43f5e", linestyle="--", lw=1.5, label=f"Macro Cut (K={MACRO_K})")
    ax_dendro.axhline(cut_k30, color="#38bdf8", linestyle="--", lw=1.5, label=f"Micro Cut (K={PRIMARY_K})")
    ax_dendro.set_title("CAH (Ward) Truncated Dendrogram (Top 40 Merges)", fontsize=11, fontweight="bold")
    ax_dendro.set_xlabel("Cluster Size / Merge Leaf", fontsize=9.5)
    ax_dendro.set_ylabel("Ward Linkage Distance", fontsize=9.5)
    leg1 = ax_dendro.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
    for t in leg1.get_texts():
        t.set_color("#e2e8f0")

    # Right: Merge Distance Curve for final 60 merges
    last_merges = Z_ward[-60:, 2][::-1]
    k_axis = np.arange(1, 61)
    ax_jump.plot(k_axis, last_merges, color="#a855f7", marker="o", markersize=4, lw=1.8, label="Ward Merge Distance")
    ax_jump.axvline(MACRO_K, color="#f43f5e", linestyle="--", lw=1.4, label=f"Macro Genre Split (K={MACRO_K})")
    ax_jump.axvline(PRIMARY_K, color="#38bdf8", linestyle="--", lw=1.4, label=f"Micro Subgenre Split (K={PRIMARY_K})")
    ax_jump.set_title("CAH Ward Linkage Jump Plot (Last 60 Merges)", fontsize=11, fontweight="bold")
    ax_jump.set_xlabel("Number of Clusters (K)", fontsize=9.5)
    ax_jump.set_ylabel("Merge Distance (Delta Inertia)", fontsize=9.5)
    leg2 = ax_jump.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", fontsize=8.5)
    for t in leg2.get_texts():
        t.set_color("#e2e8f0")

    plt.tight_layout()
    out_path = out_dir / "03_cah_ward_dendrogram_jumps.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved CAH dendrogram and jump plot to {out_path}")


def plot_umap_4panel_comparison(
    umap_2d: np.ndarray, df_assign: pd.DataFrame, out_dir: Path
) -> None:
    """
    Plots a 2x2 grid on the 2D UMAP canvas comparing how K-Means, CAH Ward,
    CAH Single (chaining), and HDBSCAN (density + noise) partition the manifold.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=180)
    apply_dark_theme(fig, axes)

    panels = [
        (axes[0, 0], f"kmeans_{PRIMARY_K}", f"K-Means (k-means++, K={PRIMARY_K}) — Voronoi Spherical Cells"),
        (axes[0, 1], f"cah_ward_{PRIMARY_K}", f"CAH (Ward Linkage, K={PRIMARY_K}) — Hierarchical Subgenres"),
        (axes[1, 0], f"cah_single_{PRIMARY_K}", f"CAH (Single Linkage, K={PRIMARY_K}) — Chaining Collapse"),
        (axes[1, 1], "hdbscan", "HDBSCAN Baseline — Density Islands + Grey Noise (-1)"),
    ]

    cmap = plt.get_cmap("tab20")

    for ax, col, title in panels:
        labels = df_assign[col].to_numpy()
        noise_mask = labels == -1

        # Plot noise points first in translucent grey (for HDBSCAN)
        if np.any(noise_mask):
            ax.scatter(
                umap_2d[noise_mask, 0],
                umap_2d[noise_mask, 1],
                s=2.5,
                c="#475569",
                alpha=0.22,
                edgecolors="none",
                label=f"Noise ({noise_mask.sum():,})",
            )

        valid_mask = ~noise_mask
        valid_labels = labels[valid_mask]
        colors = [cmap(int(lbl) % 20) for lbl in valid_labels]

        ax.scatter(
            umap_2d[valid_mask, 0],
            umap_2d[valid_mask, 1],
            s=3.5,
            c=colors,
            alpha=0.65,
            edgecolors="none",
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("UMAP Dimension 1", fontsize=9)
        ax.set_ylabel("UMAP Dimension 2", fontsize=9)

    fig.suptitle(
        "Manifold Partition Comparison on 10,000 TMDB Movies (50D Clustering Projected to 2D UMAP)",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "04_umap_4panel_algorithm_comparison.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved 4-panel UMAP comparison to {out_path}")


def plot_agreement_heatmap(df_agree: pd.DataFrame, out_dir: Path) -> None:
    """Plots a symmetric heatmap of Adjusted Rand Index (ARI) and NMI between algorithms."""
    algos = [
        f"kmeans_{PRIMARY_K}",
        f"consolidated_{PRIMARY_K}",
        f"cah_ward_{PRIMARY_K}",
        f"cah_complete_{PRIMARY_K}",
        f"cah_average_{PRIMARY_K}",
        f"cah_single_{PRIMARY_K}",
        "hdbscan",
    ]
    short_names = [
        "K-Means",
        "Consolidated",
        "CAH Ward",
        "CAH Complete",
        "CAH Average",
        "CAH Single",
        "HDBSCAN",
    ]
    n = len(algos)
    ari_mat = np.eye(n)
    nmi_mat = np.eye(n)

    for _, row in df_agree.iterrows():
        if row["algo_1"] in algos and row["algo_2"] in algos:
            i = algos.index(row["algo_1"])
            j = algos.index(row["algo_2"])
            ari_mat[i, j] = ari_mat[j, i] = row["ARI"]
            nmi_mat[i, j] = nmi_mat[j, i] = row["NMI"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=180)
    apply_dark_theme(fig, [ax1, ax2])

    for ax, mat, title in [
        (ax1, ari_mat, "Adjusted Rand Index (ARI)"),
        (ax2, nmi_mat, "Normalized Mutual Information (NMI)"),
    ]:
        ax.grid(False)
        im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(n), labels=short_names, rotation=35, ha="right")
        ax.set_yticks(range(n), labels=short_names)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                txt_color = "#0f172a" if val > 0.55 else "#f8fafc"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=8.5, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#94a3b8")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8", fontsize=8)

    fig.suptitle(f"Cross-Algorithm Partition Agreement Matrix (K = {PRIMARY_K})", color="#f8fafc", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = out_dir / "05_algorithm_agreement_heatmap.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved agreement heatmap to {out_path}")


def profile_clusters_for_algorithm(
    X_weighted: sparse.csr_matrix,
    X_pca50: np.ndarray,
    umap_2d: np.ndarray,
    feature_names: list[str],
    df_assign: pd.DataFrame,
    label_col: str,
) -> dict[str, dict]:
    """
    Automatically names and profiles every cluster in `label_col` by extracting:
    - Top 2 Genres & Top 4 Keywords/Overview terms from mean X_weighted vector
    - Medoid movie (closest film in 50D space to cluster centroid)
    - Top 3 most popular exemplar films
    - 2D UMAP visual center (median x, y) for canvas rendering
    """
    labels = df_assign[label_col].to_numpy()
    unique_labels = sorted([int(k) for k in np.unique(labels) if k != -1])

    feature_arr = np.array(feature_names)
    genre_idx = np.where(np.char.startswith(feature_arr, "genre:"))[0]
    kw_idx = np.where(
        np.char.startswith(feature_arr, "kw:") | np.char.startswith(feature_arr, "ov:")
    )[0]

    profiles: dict[str, dict] = {}

    for k in unique_labels:
        mask = labels == k
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        # 1. Mean feature profile in X_weighted
        mean_vec = np.asarray(X_weighted[indices].mean(axis=0)).ravel()

        top_g_local = genre_idx[np.argsort(mean_vec[genre_idx])[::-1][:2]]
        top_genres = [
            feature_names[i].replace("genre:", "")
            for i in top_g_local
            if mean_vec[i] > 0.01
        ]

        top_kw_local = kw_idx[np.argsort(mean_vec[kw_idx])[::-1][:4]]
        top_keywords = [
            feature_names[i].replace("kw:", "").replace("ov:", "")
            for i in top_kw_local
            if mean_vec[i] > 0.005
        ]

        genre_str = " / ".join(top_genres) if top_genres else "Mixed Genre"
        kw_str = ", ".join(top_keywords[:3]) if top_keywords else "general"
        auto_name = f"{genre_str} ({kw_str})"

        # 2. Medoid Movie (closest to 50D centroid)
        cluster_pca = X_pca50[indices]
        centroid_50d = cluster_pca.mean(axis=0)
        dists = np.linalg.norm(cluster_pca - centroid_50d, axis=1)
        medoid_global_idx = int(indices[np.argmin(dists)])
        medoid_row = df_assign.iloc[medoid_global_idx]

        # 3. Top 3 Popular Exemplars
        sub_df = df_assign.iloc[indices].sort_values("popularity", ascending=False).head(3)
        exemplars = [
            f"{r['title']} ({int(r['release_year'])})" for _, r in sub_df.iterrows()
        ]

        # 4. 2D UMAP visual center (median is robust against stray points)
        cx = float(np.median(umap_2d[indices, 0]))
        cy = float(np.median(umap_2d[indices, 1]))

        profiles[str(k)] = {
            "cluster_id": k,
            "name": auto_name,
            "size": int(len(indices)),
            "share_pct": round(100.0 * len(indices) / len(labels), 2),
            "top_genres": top_genres,
            "top_keywords": top_keywords,
            "medoid_movie": f"{medoid_row['title']} ({int(medoid_row['release_year'])})",
            "top_exemplars": exemplars,
            "cx": round(cx, 4),
            "cy": round(cy, 4),
        }

    return profiles


def run_evaluation_pipeline() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Phase 2, 3, and 4 artifacts
    X_weighted = sparse.load_npz(PROCESSED_DIR / "X_weighted.npz")
    X_pca50 = np.load(PROCESSED_DIR / "X_pca50.npy")
    umap_2d = np.load(PROCESSED_DIR / "umap_2d.npy")
    with open(PROCESSED_DIR / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    df_metrics = pd.read_csv(BENCHMARK_DIR / "metrics_grid.csv")
    df_agree = pd.read_csv(BENCHMARK_DIR / "algorithm_agreement.csv")
    df_assign = pd.read_parquet(BENCHMARK_DIR / "cluster_assignments.parquet")
    Z_ward = np.load(BENCHMARK_DIR / "cah_ward_linkage.npy")

    # 2. Generate all 4 comparative study figures
    print("--- Generating Comparative Study Figures ---")
    plot_internal_metrics_curves(df_metrics, FIGURES_DIR)
    plot_cah_dendrogram_and_jumps(Z_ward, FIGURES_DIR)
    plot_umap_4panel_comparison(umap_2d, df_assign, FIGURES_DIR)
    plot_agreement_heatmap(df_agree, FIGURES_DIR)

    # 3. Automated Semantic Cluster Naming across primary algorithms
    print("\n--- Profiling & Naming Clusters ---")
    target_Label_cols = [
        f"kmeans_{PRIMARY_K}",
        f"consolidated_{PRIMARY_K}",
        f"cah_ward_macro_{MACRO_K}",
        f"cah_ward_{PRIMARY_K}",
        "hdbscan",
    ]

    all_profiles: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for col in target_Label_cols:
        if col not in df_assign.columns:
            continue
        prof = profile_clusters_for_algorithm(
            X_weighted, X_pca50, umap_2d, feature_names, df_assign, col
        )
        all_profiles[col] = prof
        for k_str, meta in prof.items():
            summary_rows.append(
                {
                    "algorithm": col,
                    "cluster_id": meta["cluster_id"],
                    "size": meta["size"],
                    "share_pct": meta["share_pct"],
                    "auto_label": meta["name"],
                    "medoid_movie": meta["medoid_movie"],
                    "exemplars": " | ".join(meta["top_exemplars"]),
                }
            )

    # Save semantic profiles JSON (for Phase 6 web frontend) and CSV (for report tables)
    profiles_json_path = BENCHMARK_DIR / "cluster_profiles.json"
    with open(profiles_json_path, "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, indent=2)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(BENCHMARK_DIR / "cluster_profiles_summary.csv", index=False)

    print(f"\n=== Sample Semantic Cluster Profiles (Consolidated CAH+KMeans K={PRIMARY_K}, Top 10 by Size) ===")
    top10 = (
        df_summary[df_summary["algorithm"] == f"consolidated_{PRIMARY_K}"]
        .sort_values("size", ascending=False)
        .head(10)
    )
    print(
        top10[["cluster_id", "size", "share_pct", "auto_label", "medoid_movie"]].to_string(
            index=False
        )
    )
    print(f"\nSaved cluster profiles to {profiles_json_path}")


if __name__ == "__main__":
    run_evaluation_pipeline()
