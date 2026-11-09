import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, normalize
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"



def load_weighted_matrix(processed_dir: Path = PROCESSED_DIR) -> sparse.csr_matrix:
    """Loads the block-weighted sparse matrix from Phase 2."""
    matrix_path = processed_dir / "X_weighted.npz"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Matrix not found at {matrix_path}. Run preprocess.py first."
        )
    X = sparse.load_npz(matrix_path)
    print(f"Loaded X_weighted of shape {X.shape}")
    return X


def build_clustering_space(
    X: sparse.csr_matrix, n_components: int = 50, random_state: int = 42
) -> tuple[np.ndarray, dict]:
    """
    Reduces the sparse feature matrix to a dense 50D latent space via TruncatedSVD,
    then re-applies L2 normalization so Euclidean distance remains monotonic with Cosine distance.
    """
    print(f"Running TruncatedSVD to {n_components} dimensions...")
    svd = TruncatedSVD(
        n_components=n_components, algorithm="randomized", n_iter=7, random_state=random_state
    )
    X_svd = svd.fit_transform(X).astype(np.float32)

    explained_var = float(np.sum(svd.explained_variance_ratio_))
    print(
        f"TruncatedSVD ({n_components}D) cumulative explained variance: {explained_var:.2%}"
    )

    # Re-normalize rows to unit length on the 50D hypersphere
    X_pca50 = normalize(X_svd, norm="l2", axis=1).astype(np.float32)

    svd_meta = {
        "n_components": n_components,
        "cumulative_explained_variance": round(explained_var, 4),
        "top_10_singular_values": [round(float(v), 4) for v in svd.singular_values_[:10]],
        "top_10_variance_ratios": [
            round(float(v), 4) for v in svd.explained_variance_ratio_[:10]
        ],
    }
    return X_pca50, svd_meta


def build_umap_canvas_space(
    X_pca50: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Projects the 50D latent vectors onto a 2D manifold using UMAP for visualization,
    scaling coordinates into [-1, 1] for the HTML5 Canvas.
    """
    print(
        f"Running 2D UMAP projection (n_neighbors={n_neighbors}, min_dist={min_dist})..."
    )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=None,
        low_memory=False,
    )
    coords_2d = reducer.fit_transform(X_pca50).astype(np.float32)

    # Normalize coordinates to [-1, 1] for canvas rendering
    scaler = MinMaxScaler(feature_range=(-1, 1))
    coords_normalized = scaler.fit_transform(coords_2d).astype(np.float32)

    print(f"Generated 2D UMAP coordinates of shape {coords_normalized.shape}")
    return coords_normalized


def run_reduction_pipeline(
    n_components: int = 50, umap_neighbors: int = 30, umap_min_dist: float = 0.1
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_weighted = load_weighted_matrix(PROCESSED_DIR)

    # 1. Build 50D Clustering Space
    X_pca50, svd_meta = build_clustering_space(X_weighted, n_components=n_components)
    np.save(PROCESSED_DIR / "X_pca50.npy", X_pca50)

    with open(PROCESSED_DIR / "svd_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(svd_meta, f, indent=2)

    # 2. Build 2D UMAP Visualization Space
    umap_2d = build_umap_canvas_space(
        X_pca50, n_neighbors=umap_neighbors, min_dist=umap_min_dist
    )
    np.save(PROCESSED_DIR / "umap_2d.npy", umap_2d)

    print(f"Saved X_pca50.npy and umap_2d.npy to {PROCESSED_DIR}")


def plot_umap_diagnostic(
    processed_dir: Path = PROCESSED_DIR,
    output_path: Path = FIGURES_DIR / "01_umap_genre_sanity_check.png",
) -> None:
    """
    Plots the 2D UMAP coordinates colored by each movie's primary genre
    to verify manifold topology prior to Phase 4 clustering.
    """
    umap_2d = np.load(processed_dir / "umap_2d.npy")
    df = pd.read_parquet(processed_dir / "movies_index.parquet")

    # Extract the first listed genre as the primary genre for coloring
    df["primary_genre"] = (
        df["genres"]
        .fillna("Unknown")
        .apply(lambda s: s.split(",")[0].strip() if s else "Unknown")
    )

    # Keep top 10 most frequent primary genres; group the rest as 'Other'
    top_genres = df["primary_genre"].value_counts().nlargest(10).index.tolist()
    df["genre_group"] = df["primary_genre"].where(
        df["primary_genre"].isin(top_genres), "Other"
    )

    palette = [
        "#38bdf8", "#f43f5e", "#facc15", "#a855f7", "#4ade80",
        "#fb923c", "#2dd4bf", "#e879f9", "#60a5fa", "#f87171", "#64748b"
    ]
    groups = top_genres + (["Other"] if "Other" in df["genre_group"].values else [])

    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=180)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    for idx, genre in enumerate(groups):
        mask = (df["genre_group"] == genre).to_numpy()
        color = palette[idx % len(palette)]
        ax.scatter(
            umap_2d[mask, 0],
            umap_2d[mask, 1],
            s=4.5,
            c=color,
            alpha=0.55 if genre != "Other" else 0.25,
            edgecolors="none",
            label=f"{genre} ({mask.sum():,})",
        )

        # Place label at the median coordinate of each top genre
        if genre != "Other" and mask.sum() > 50:
            med_x = float(np.median(umap_2d[mask, 0]))
            med_y = float(np.median(umap_2d[mask, 1]))
            ax.text(
                med_x,
                med_y,
                genre,
                color="white",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor="#1e293b",
                    edgecolor=color,
                    alpha=0.88,
                    lw=1.1,
                ),
            )

    ax.set_title(
        f"TMDB 2D UMAP Manifold ({len(df):,} Movies Colored by Primary Genre)",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("UMAP Dimension 1", color="#94a3b8", fontsize=10)
    ax.set_ylabel("UMAP Dimension 2", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(True, color="#1e293b", linestyle="--", linewidth=0.6, alpha=0.7)

    leg = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="#1e293b",
        edgecolor="#334155",
        fontsize=8.5,
        markerscale=3,
        title="Primary Genre",
        title_fontsize=9,
    )
    leg.get_title().set_color("#f8fafc")
    for text in leg.get_texts():
        text.set_color("#e2e8f0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved UMAP diagnostic visualization to {output_path}")


if __name__ == "__main__":
    run_reduction_pipeline()
    plot_umap_diagnostic()