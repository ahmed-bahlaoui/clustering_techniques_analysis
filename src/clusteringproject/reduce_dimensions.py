import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, normalize
import umap

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


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
        random_state=random_state,
        low_memory=False,
    )
    coords_2d = reducer.fit_transform(X_pca50).astype(np.float32)

    # Normalize coordinates to [-1, 1] for canvas rendering
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
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


if __name__ == "__main__":
    run_reduction_pipeline()