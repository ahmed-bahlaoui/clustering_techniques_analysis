# Complete Pipeline & File-by-File Architecture Guide

This document provides an exhaustive, file-by-file breakdown of the **`clustering_techniques_analysis`** project. It traces the exact lifecycle of the data from a raw $930,000+$-row CSV on Kaggle down to the synchronized mathematical matrices and lookup tables used for clustering and interactive visualization.

---

## 1. High-Level Data Flow Overview

Our pipeline separates **human-readable metadata** (strings like movie titles and poster URLs) from **mathematical feature matrices** (numbers used by clustering algorithms). Every output file in `data/processed/` contains the exact same $10,000$ movies in the exact same row order (Row `0` to Row `9,999`).

```text
[Kaggle Raw CSV: ~930,000+ rows × 24 columns]
       │
       ▼  1. src/clusteringproject/fetch_data.py
[data/movies.db: 10,000 rows × 12 columns (SQLite)]
       │
       ▼  2. src/clusteringproject/preprocess.py
       ├──► data/processed/movies_index.parquet  (10,000 × 6)   [Human Lookup Table]
       ├──► data/processed/feature_names.json    (524 strings)  [Column Dictionary]
       └──► data/processed/X_weighted.npz        (10,000 × 524) [Sparse Feature Matrix]
                   │
                   ▼  3. src/clusteringproject/reduce_dimensions.py
                   ├──► data/processed/X_pca50.npy          (10,000 × 50) [Clustering Input Matrix]
                   ├──► data/processed/svd_diagnostics.json (Metadata)    [Variance Explained]
                   ├──► data/processed/umap_2d.npy          (10,000 × 2)  [2D Screen Coordinates]
                   └──► reports/figures/01_umap_genre_sanity_check.png    [Diagnostic Plot]
```

---

## 2. Phase 1: Data Ingestion & Filtering (`src/clusteringproject/fetch_data.py`)

### Purpose of the File
`fetch_data.py` downloads the raw TMDB dataset from Kaggle, strips out irrelevant columns and noisy/obscure entries, and saves the top $10,000$ feature-length movies into a fast, indexed local SQLite database (`data/movies.db`).

### A. The Starting Data (Raw Kaggle CSV)
We pull the dataset `asaniczka/tmdb-movies-dataset-2023-930k-movies` using `kagglehub`. The raw CSV contains over $930,000$ rows (including unreleased projects, short films, student videos, and adult content) and **24 columns**:

1. `id`
2. `title`
3. `vote_average`
4. `vote_count`
5. `status`
6. `release_date`
7. `revenue`
8. `runtime`
9. `adult`
10. `backdrop_path`
11. `budget`
12. `homepage`
13. `imdb_id`
14. `original_language`
15. `original_title`
16. `overview`
17. `popularity`
18. `poster_path`
19. `tagline`
20. `genres`
21. `production_companies`
22. `production_countries`
23. `spoken_languages`
24. `keywords`

### B. Columns Removed in Phase 1 (and Why)
During `download_and_load_raw_csv()` and `clean_and_filter_movies()`, we discard **12 columns**:

| Removed Column | Why We Removed It |
| :--- | :--- |
| `budget`, `revenue` | Over $70\%$ of movies in TMDB report `$0` for budget and revenue. Including them would introduce massive zero-inflation noise or force us to drop classic/indie films. |
| `imdb_id`, `homepage`, `backdrop_path` | External URLs and identifiers that carry zero thematic or mathematical signal for clustering. |
| `original_title` | Redundant with `title` and `original_language`. |
| `tagline` | Often missing or consists of short promotional puns rather than actual plot descriptions (unlike `overview`). |
| `production_companies`, `production_countries`, `spoken_languages` | Studio names (`Warner Bros.`, `Universal`) group movies by corporate distributor rather than thematic content, which would distort genre clusters. |
| `adult`, `status` | Used **only as temporary filter columns** (`adult == False` and `status == 'Released'`) and then dropped because every remaining row has the exact same value. |

### C. Row Filtering Rules (Why We Keep $N = 10,000$ Rows)
In `clean_and_filter_movies()`, we apply five strict filters:
1. **Non-Adult & Released Only:** Keeps only legitimate, released cinema (`adult == False`, `status == 'Released'`).
2. **Valid Core Metadata:** Drops rows missing `id`, `title`, `genres`, or `release_date`.
3. **Vote Threshold (`vote_count >= 50`):** Removes obscure entries with unreliable ratings (e.g., 1 vote of 10.0) and missing community keywords.
4. **Feature-Length Runtime (`40 <= runtime <= 300`):** Excludes 3-minute YouTube shorts and 10-hour television miniseries.
5. **Top $10,000$ by `vote_count`:** Standard Hierarchical Clustering (CAH) requires $\mathcal{O}(N^2)$ memory to store its pairwise distance matrix. At $N = 10,000$, the distance matrix takes $\sim 400\text{--}800\text{ MB}$ of RAM, running smoothly on any machine while capturing the $10,000$ most culturally significant films.

### D. Columns Kept in `data/movies.db` (12 Columns)
We transform `release_date` (e.g., `"2010-07-16"`) into an integer `release_year` (`2010`) and save **12 clean columns** into the `movies` table inside `data/movies.db`:
* **Identifiers & UI Display (4 columns):** `id`, `title`, `original_language`, `poster_path`
* **Continuous Numerical Metadata (5 columns):** `release_year`, `runtime`, `vote_average`, `vote_count`, `popularity`
* **Text & Categorical Features (3 columns):** `genres`, `keywords`, `overview`

---

## 3. Phase 2: Feature Engineering & Vectorization (`src/clusteringproject/preprocess.py`)

### Purpose of the File
Clustering algorithms like K-Means and CAH cannot process raw strings (`"Action, Sci-Fi"`) or unscaled numbers (`vote_count = 35000` vs. `vote_average = 7.8`). `preprocess.py` loads the $10,000$ movies from `data/movies.db` (sorted by `id ASC` to lock row order) and converts their thematic and numerical features into a single unified mathematical matrix (`X_weighted.npz`).

### How `build_feature_matrix()` Works Step-by-Step

1. **Genres Block (`MultiLabelBinarizer` $\to \sim 19$ columns):**
   * Splits comma-separated genres into binary $0/1$ columns (`genre:Action`, `genre:Horror`, etc.).
2. **Keywords Block (`TfidfVectorizer` $\to 300$ columns):**
   * Splits comma-separated keywords (`kw:time travel`, `kw:dystopia`), keeps the top $300$ keywords appearing in at least $5$ movies, and applies **TF-IDF weighting** so rare, specific tropes outweigh generic tags.
3. **Overview Block (`TfidfVectorizer` $\to 200$ columns):**
   * Extracts the top $200$ English unigrams and bigrams (`ov:world war`, `ov:high school`) from the `overview` plot summary (appearing in at least $10$ movies and at most $50\%$ of movies) to act as a fallback when community keywords are sparse.
4. **Numerical Metadata Block (`StandardScaler` $\to 5$ columns):**
   * Takes `release_year`, `runtime`, `vote_average`, $\log(1 + \text{vote\_count})$, and $\log(1 + \text{popularity})$, and standardizes each column to mean $\mu = 0$ and standard deviation $\sigma = 1$.
5. **Dimensionality-Adjusted Block Weighting ($w_b / \sqrt{d_b}$):**
   * Because `keywords` ($300$ cols) and `overview` ($200$ cols) have far more columns than `genres` ($19$ cols), raw concatenation would cause text columns to dominate Euclidean distance. Each block $X_b$ is scaled by its domain weight divided by the square root of its column count:
     $$\tilde{X}_b = X_b \cdot \frac{w_b}{\sqrt{d_b}}$$
     using weights $w_{\text{genres}} = 2.5$, $w_{\text{keywords}} = 2.0$, $w_{\text{overview}} = 1.0$, and $w_{\text{numeric}} = 0.6$.
6. **Row-Wise L2 Normalization (`normalize(..., norm='l2')`):**
   * Divides every movie's $524$-dimensional row vector by its Euclidean length so $\Vert\mathbf{x}_i\Vert_2 = 1$. On a unit hypersphere, squared Euclidean distance is directly proportional to Cosine distance:
     $$\Vert\mathbf{u} - \mathbf{v}\Vert_2^2 = 2(1 - \cos(\mathbf{u}, \mathbf{v}))$$
     This eliminates bias toward famous movies that have more tags and makes Euclidean K-Means and Ward's CAH sensitive to thematic angles.

### Outputs Generated by `preprocess.py` (in `data/processed/`)

| Output File | Shape / Type | What It Contains & Why We Keep It |
| :--- | :--- | :--- |
| **`X_weighted.npz`** | `(10000, ~524)` sparse `float32` | **The Full Mathematical Feature Matrix.** Contains all encoded genres, keywords, overview terms, and numeric metadata. Passed into Phase 3 for dimensionality reduction and used in Phase 5 to automatically name clusters. |
| **`feature_names.json`** | List of `~524` strings | **The Column Dictionary.** Maps every column index in `X_weighted.npz` to its human name (e.g., `"genre:Horror"`, `"kw:slasher"`). Used in Phase 5 to inspect cluster centroids and print their top defining keywords. |
| **`movies_index.parquet`** | `(10000, 6)` DataFrame | **The Human Lookup Table.** Contains only `id`, `title`, `release_year`, `genres`, `vote_average`, and `popularity`. Why only 6 columns? Because heavy text columns (`overview`, `keywords`) are already mathematically encoded inside `X_weighted.npz`. Dropping the raw text paragraphs makes this alignment table tiny, fast to load in Pandas, and ready for plotting and web export. |

---

## 4. Phase 3: Dimensionality Reduction (`src/clusteringproject/reduce_dimensions.py`)

### Purpose of the File
`reduce_dimensions.py` takes the $524$-dimensional sparse matrix `X_weighted.npz` and performs the **Dual-Space Split**, creating two distinct dense matrices: one optimized for clustering algorithms ($50\text{D}$) and one optimized for human visual inspection on a screen ($2\text{D}$).

### How `reduce_dimensions.py` Works Step-by-Step

1. **`build_clustering_space()` $\to$ `X_pca50.npy` ($10,000 \times 50$):**
   * **Why not cluster directly on `X_weighted.npz` ($524\text{D}$)?** In $524$ sparse dimensions, Euclidean distances suffer from the *curse of dimensionality* (distances between points converge), and synaptic synonym noise (e.g., one movie tagged `kw:alien` and another `ov:extraterrestrial`) artificially separates similar films.
   * **How it works:** Runs `TruncatedSVD(n_components=50)` (Latent Semantic Analysis) to compress the $524$ sparse columns into the top $50$ dense orthogonal latent components, then re-applies row-wise **L2 normalization** (`normalize(X_svd, norm='l2')`) so every movie vector sits back on the $50\text{D}$ unit hypersphere.
   * **Role in the Project:** **`X_pca50.npy` is the exact mathematical dataset fed into K-Means, CAH, and HDBSCAN in Phase 4.**

2. **`build_umap_canvas_space()` $\to$ `umap_2d.npy` ($10,000 \times 2$):**
   * **Why not cluster directly on 2D UMAP?** UMAP nonlinearly compresses and tears global distances to pack points onto a flat 2D plane. Running K-Means or Ward's CAH on 2D UMAP coordinates measures distorted visual artifacts rather than true high-dimensional structure.
   * **How it works:** Runs `umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='cosine')` on `X_pca50.npy`, followed by `MinMaxScaler(feature_range=(-1.0, 1.0))` so all $(x, y)$ coordinates fall cleanly between $-1$ and $+1$.
   * **Role in the Project:** Used strictly as $(x, y)$ screen coordinates for static plots and the Phase 6 HTML5 Canvas web app.

3. **`plot_umap_diagnostic()` $\to$ `reports/figures/01_umap_genre_sanity_check.png`:**
   * **How it works:** Loads the $(x, y)$ coordinates from `umap_2d.npy` and the `genres` string from `movies_index.parquet`, extracts each movie's first listed genre (`df["genres"].split(",")[0]`), and plots all $10,000$ movies colored by their primary genre.
   * **Role in the Project:** Validates that Phases 1–3 successfully separated distinct genres into high-density islands and continuous crossover bridges before running any clustering algorithm.

---

## 5. Summary Table: Repository Files & Artifacts Inventory

| File Path | Stage | Role |
| :--- | :--- | :--- |
| `src/clusteringproject/fetch_data.py` | Phase 1 (Code) | Downloads Kaggle CSV, filters rows/columns, builds `data/movies.db`. |
| `data/movies.db` | Phase 1 (Data) | Master SQLite database ($10,000$ rows $\times 12$ cleaned columns). |
| `src/clusteringproject/preprocess.py` | Phase 2 (Code) | Vectorizes text/tags/numbers, applies block weights & L2 normalization. |
| `data/processed/X_weighted.npz` | Phase 2 (Data) | Sparse $(10000, 524)$ block-weighted feature matrix. |
| `data/processed/feature_names.json` | Phase 2 (Data) | List of all $524$ feature names for cluster profiling in Phase 5. |
| `data/processed/movies_index.parquet` | Phase 2 (Data) | Fast $(10000, 6)$ lookup table (`id`, `title`, `release_year`, `genres`, `vote_average`, `popularity`). |
| `src/clusteringproject/reduce_dimensions.py` | Phase 3 (Code) | Runs `TruncatedSVD` ($50\text{D}$), `UMAP` ($2\text{D}$), and generates the sanity-check plot. |
| `data/processed/X_pca50.npy` | Phase 3 (Data) | **Primary Clustering Dataset:** Dense $(10000, 50)$ L2-normalized latent matrix. |
| `data/processed/svd_diagnostics.json` | Phase 3 (Data) | Records explained variance ratio and top singular values of the $50\text{D}$ SVD. |
| `data/processed/umap_2d.npy` | Phase 3 (Data) | **Primary Visualization Coordinates:** Dense $(10000, 2)$ array scaled to $[-1, 1]$. |
| `reports/figures/01_umap_genre_sanity_check.png` | Phase 3 (Plot) | 2D UMAP scatter plot colored by primary genre to verify manifold topology. |