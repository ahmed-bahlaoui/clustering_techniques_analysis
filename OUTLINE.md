# Comparative Study of Unsupervised Clustering on High-Dimensional Cinema Data
**Inspired by Mikey Guinness’s Interactive Movie Map**

---

## 1. Executive Summary & Project Goal

### The Inspiration
Mikey Guinness’s **Movie Map** visualizes thousands of films from The Movie Database (TMDB) as an interactive 2D galaxy, where proximity reflects thematic similarity. His original pipeline projects a ~600-dimensional metadata space into 2D using **UMAP** and identifies dense genre islands using **HDBSCAN** (a density-based clustering algorithm).

### Our Research Objective
While density-based clustering (HDBSCAN) excels at rejecting noise, it leaves a large percentage of transitional or arthouse films unclustered and produces a flat partition. 

The goal of this project is to **re-architect the Movie Map as a rigorous comparative study** evaluating three distinct clustering paradigms on the exact same latent cinema space ($N = 10,000$ films):
1. **Centroid-Based Partitioning:** **K-Means** (`k-means++`)
2. **Hierarchical Agglomerative Clustering:** **CAH** (*Classification Ascendante Hiérarchique*) across multiple linkage criteria (Ward, Complete, Average, Single)
3. **Hybrid Consolidation:** **CAH + K-Means Consolidation** (initializing K-Means with hierarchical barycenters)
4. **Density-Based Baseline:** **HDBSCAN** (the original baseline)

---

## 2. End-to-End Architecture & Pipeline Structure

The project is divided into six modular stages inside `src/clusteringproject/`:

```text
[Kaggle TMDB Dataset (930k+ films)]
        │
        ▼  Phase 1: fetch_data.py
[SQLite Database: Top 10,000 Films (data/movies.db)]
        │
        ▼  Phase 2: preprocess.py
[Block-Weighted & L2-Normalized Matrix X (10,000 × 524)]
        │
        ▼  Phase 3: reduce_dimensions.py (The Dual-Space Split)
        ├───────────────────────────────────────────┐
        ▼                                           ▼
[Clustering Space: 50D TruncatedSVD/PCA]    [Visual Space: 2D UMAP Coordinates]
        │                                           │
        ▼  Phase 4: cluster_benchmark.py            │
[K-Means | CAH (Ward/Avg/Single) | Hybrid | HDBSCAN]│
        │                                           │
        ▼  Phase 5: evaluate_and_profile.py         │
[Metrics (Silhouette, DBI, ARI) + TF-IDF Naming]    │
        │                                           │
        └───────────────────┬───────────────────────┘
                            ▼  Phase 6: export_json.py & Web App
            [Interactive HTML5 Canvas Comparator]
```

### Key Methodological Decisions
* **Why $N = 10,000$ movies?** Standard CAH requires computing and storing a pairwise distance matrix of complexity $\mathcal{O}(N^2)$ in memory and $\mathcal{O}(N^3)$ in time. Filtering to the top 10,000 films by `vote_count` keeps the distance matrix under $800\text{ MB}$ while capturing all culturally significant films.
* **Why Block-Weighted Vectorization?** Raw concatenation of ~19 genres, 300 plot keywords, 200 overview terms, and 5 numerical features would cause the 300 keywords to dominate Euclidean distance. We scale each block $b$ by $w_b / \sqrt{d_b}$:
  $$\tilde{X}_b = X_b \cdot \frac{w_b}{\sqrt{d_b}}$$
  where $w_{\text{genres}} = 2.5$, $w_{\text{keywords}} = 2.0$, $w_{\text{overview}} = 1.0$, and $w_{\text{numeric}} = 0.6$.
* **Why Row-Wise L2 Normalization?** Normalizing every film vector to unit length ($\|\mathbf{x}_i\|_2 = 1$) ensures that squared Euclidean distance is directly proportional to Cosine distance:
  $$\|\mathbf{x}_i - \mathbf{x}_j\|_2^2 = 2\left(1 - \cos(\mathbf{x}_i, \mathbf{x}_j)\right)$$
  This allows K-Means and Ward’s CAH (which require Euclidean geometry) to cluster movies by **thematic angle** rather than total tag count.
* **Why the Dual-Space Split ($50\text{D}$ vs. $2\text{D}$)?** Clustering directly on $524\text{D}$ sparse data suffers from the curse of dimensionality, while clustering on $2\text{D}$ UMAP coordinates introduces nonlinear distortion. We cluster in the **top 50 linear SVD components** ($X_{50\text{D}}$) and reserve $2\text{D}$ UMAP strictly for visual rendering.

---

## 3. Comparative Analysis Framework

### Algorithmic Comparison Matrix

| Analytical Dimension | K-Means (`k-means++`) | CAH (Agglomerative Hierarchical) | Hybrid: CAH + K-Means Consolidation | HDBSCAN (Original Baseline) |
| :--- | :--- | :--- | :--- | :--- |
| **Objective Function** | Minimizes within-cluster inertia ($W$) via iterative centroid updates | Greedily merges closest clusters bottom-up based on linkage criterion | Uses CAH Ward barycenters as initial seeds for K-Means refinement | Extracts stable high-density islands from Mutual Reachability MST |
| **Cluster Geometry** | Convex, isotropic Voronoi cells (spherical) | **Ward:** Spherical<br>**Average/Complete:** Compact semantic cliques<br>**Single:** Elongated chains | Convex Voronoi cells anchored on hierarchical structure | Arbitrary, non-convex shapes of variable density |
| **Handling of Crossover & Outlier Films** | **Forced Assignment:** Pulls experimental/arthouse films into the nearest mainstream cluster | **Forced Assignment:** Every film belongs to a branch at cut height $K$ | **Forced Assignment:** Reassigns boundary films to optimize global inertia | **Noise Rejection:** Labels transitional or bizarre films as unclustered noise (`-1`) |
| **Structural Output** | Flat partition for a fixed $K$ | **Multi-scale Dendrogram:** Maps Macro-Genres ($K=12$) $\to$ Subgenres ($K=40$) | Optimized flat partition at cut $K$ | Flat density clusters + outlier scores |
| **Computational Complexity** | $\mathcal{O}(N \cdot K \cdot I \cdot d)$ (Fast, linear in $N$) | $\mathcal{O}(N^2)$ memory, $\mathcal{O}(N^3)$ time (Bottleneck at large $N$) | $\mathcal{O}(N^2)$ (dominated by CAH step) | $\mathcal{O}(N \log N)$ in low-D, $\mathcal{O}(N^2)$ in high-D |

---

### Core Hypotheses Tested in Our Study

1. **The Crossover Chaining Effect in CAH (Linkage Sensitivity):**
   * *Hypothesis:* In cinema data, multi-genre crossover films (e.g., *Action-Comedy* or *Sci-Fi Horror*) act as bridges between dense genre islands. Under **Single Linkage**, CAH will suffer from the *chaining phenomenon*, merging almost the entire film galaxy into one giant cluster while isolating single outliers. **Ward’s Linkage** and **Average Cosine Linkage** will resist chaining and produce balanced genre taxonomies.
2. **The Forced-Assignment Penalty (K-Means vs. HDBSCAN):**
   * *Hypothesis:* Because K-Means partitions the entire space into Voronoi cells, niche/experimental films will exhibit near-zero or negative individual Silhouette scores ($s_i \le 0$), whereas core genre films near centroids will match HDBSCAN’s high-density islands (high Adjusted Rand Index on non-noise points).
3. **Superiority of Hybrid Consolidation (CAH $\to$ K-Means):**
   * *Hypothesis:* CAH makes greedy, irrevocable merges early in the tree, leading to suboptimal boundary assignments at the top of the dendrogram. Initializing K-Means with CAH Ward barycenters (`Consolidation`) will achieve a lower Within-Cluster Sum of Squares ($W$) and higher Silhouette score than both standalone CAH and randomly initialized `k-means++`.

---

## 4. Quantitative & Qualitative Evaluation Metrics

To compare the algorithms objectively across $K \in \{10, 15, 20, 30, 45, 60\}$, we measure:

### 1. Internal Validation Metrics (No Ground Truth Required)
* **Within-Cluster Inertia ($W$):**
  $$W = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$$
  Used to plot the **Elbow Curve** (K-Means) and quantify the variance reduction gained via **CAH + K-Means Consolidation**.
* **Silhouette Coefficient ($S \in [-1, 1]$):**
  $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$
  Measures intra-cluster cohesion ($a_i$) against nearest-neighbor cluster separation ($b_i$). Higher is better.
* **Davies-Bouldin Index (DBI):** Measures the average similarity between each cluster and its most similar competitor. Lower is better.
* **Calinski-Harabasz Index (CHI):** Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better.

### 2. Cross-Algorithm Agreement Metrics
* **Adjusted Rand Index (ARI)** & **Normalized Mutual Information (NMI):** Quantifies how closely K-Means, CAH (Ward), and HDBSCAN agree on the underlying movie groupings.

### 3. Qualitative & Domain Interpretability
* **Automated TF-IDF Cluster Profiling:** Extracting the top 3 genres and top 5 distinctive plot keywords for every cluster centroid.
* **Archetype (Medoid) Extraction:** Identifying the real movie closest to each cluster’s center ($\arg\min_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2$).

---

## 5. Recommended Presentation Deck Outline (10 Slides)

Use this structure to build your presentation slides:

* **Slide 1 — Title & Vision:** *Mapping the Cinema Universe: A Comparative Study of K-Means, Hierarchical Clustering (CAH), and HDBSCAN.* (Include a screenshot of Mikey Guinness's Movie Map).
* **Slide 2 — Problem Statement:** How do we automatically organize 10,000+ movies into meaningful genres and micro-tropes using unsupervised learning? Why compare centroid, hierarchical, and density paradigms?
* **Slide 3 — Data Pipeline & Block-Weighted Vectorization:** Explain the Kaggle TMDB dataset, SQLite filtering, TF-IDF keyword extraction, block weighting ($w_b / \sqrt{d_b}$), and L2 normalization.
* **Slide 4 — The Dual-Space Architecture ($50\text{D}$ SVD vs. $2\text{D}$ UMAP):** Explain why we cluster in $50\text{D}$ linear latent space and visualize in $2\text{D}$ UMAP space.
* **Slide 5 — Algorithm 1: K-Means Partitioning:** Show the Elbow curve, Voronoi geometry, and how K-Means creates clean archetypal centroids but struggles with forced outlier assignment.
* **Slide 6 — Algorithm 2: CAH & The Dendrogram Taxonomy:** Compare Ward, Average, and Single linkage. Show the dendrogram and how cutting at $K=12$ (Macro-Genres) vs. $K=36$ (Subgenres) reveals a natural hierarchy.
* **Slide 7 — The Hybrid Approach (CAH + K-Means Consolidation):** Explain the two-step consolidation process and show the quantitative drop in inertia ($W$) and boost in Silhouette score.
* **Slide 8 — Quantitative Benchmark Results:** Present the comparison tables and curves (Silhouette, Davies-Bouldin, Calinski-Harabasz, and ARI heatmap across algorithms).
* **Slide 9 — Qualitative Cluster Profiles (Case Studies):** Show 3–4 discovered clusters with their automated TF-IDF tags, medoid movies, and how K-Means vs. HDBSCAN handled boundary crossover films (e.g., *Parasite* or *Everything Everywhere All at Once*).
* **Slide 10 — Live Interactive Demo & Conclusion:** Demonstrate the custom HTML5 Canvas web app switching live between K-Means, CAH hierarchy levels, and HDBSCAN.