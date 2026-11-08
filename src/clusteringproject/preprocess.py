import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize
from typing import List, cast

# Path configurations
ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "data" / "movies.db"
DB_PATH_CLEANED =  ROOT_DIR / "data" / "movies_cleaned.db"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def load_movies_from_sqlite(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Loads the cleaned movies table from SQLite."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run fetch_data.py first."
        )
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM movies ORDER BY id ASC", conn)
    print(f"Loaded {len(df):,} movies from {db_path.name}")
    return df


def tokenizer(s: str, convert_to_lowercase: bool = True) -> List[str]:
    """ this function takes a string input containing
        comma-separated values and returns a list of cleaned, strings converted or not to lowecase.
    """
    result = []
    for element in s.split(","):
        if element.strip():
            if  convert_to_lowercase:
                result.append(element.strip().lower())
            else:
                result.append(element.strip())

    return result




def build_feature_matrix(
    df: pd.DataFrame,
    max_keywords: int = 300,
    max_overview_terms: int = 200,
    weights: dict[str, float] | None = None,
) -> tuple[sparse.csr_matrix, List[str]]:
    """
    Constructs a block-weighted, L2-normalized feature matrix from movie metadata.
    """
    if weights is None:
        weights = {
            "genres": 2.5,
            "keywords": 2.0,
            "overview": 1.0,
            "numeric": 0.6,
        }

    # 1. Genres Multi-Hot Encoding
    genre_lists = (
        df["genres"]
        .fillna("")
        .apply(lambda s: tokenizer(s, convert_to_lowercase=False))
    )
    mlb_genres = MultiLabelBinarizer(sparse_output=True)
    X_genres = mlb_genres.fit_transform(genre_lists).astype(np.float32) # type: ignore
    genre_names = [f"genre:{g}" for g in mlb_genres.classes_]

    # 2. Keywords TF-IDF (comma-separated phrases)
    tfidf_kw = TfidfVectorizer(
        tokenizer=tokenizer,
        token_pattern=cast(str, None),
        max_features=max_keywords,
        min_df=5,
        dtype=np.float32,
    )
    X_keywords = tfidf_kw.fit_transform(df["keywords"].fillna(""))
    kw_names = [f"kw:{k}" for k in tfidf_kw.get_feature_names_out()]

    # 3. Overview TF-IDF (unigrams + bigrams for plot fallback)
    tfidf_ov = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_overview_terms,
        min_df=10,
        max_df=0.5,
        dtype=np.float32,
    )
    X_overview = tfidf_ov.fit_transform(df["overview"].fillna(""))
    ov_names = [f"ov:{t}" for t in tfidf_ov.get_feature_names_out()]

    # 4. Continuous Numerical Features (Log-transformed & Standardized)
    num_df = pd.DataFrame(
        {
            "release_year": df["release_year"].astype(float),
            "runtime": df["runtime"].astype(float),
            "vote_average": df["vote_average"].astype(float),
            "log_vote_count": np.log1p(df["vote_count"].astype(float)),
            "log_popularity": np.log1p(df["popularity"].astype(float)),
        }
    )
    scaler = StandardScaler()
    X_numeric = sparse.csr_matrix(scaler.fit_transform(num_df).astype(np.float32))
    num_names = [f"num:{c}" for c in num_df.columns]

    # 5. Apply Block Weighting: w_b / sqrt(d_b) so block dimensionality doesn't skew distances
    def scale_block(mat: sparse.csr_matrix, w: float) -> sparse.csr_matrix:
        n_cols = max(mat.shape[1], 1)
        return mat * (w / np.sqrt(n_cols))

    X_combined = sparse.hstack(
        [
            scale_block(X_genres, weights["genres"]),
            scale_block(X_keywords, weights["keywords"]),
            scale_block(X_overview, weights["overview"]),
            scale_block(X_numeric, weights["numeric"]),
        ],
        format="csr",
        dtype=np.float32,
    )

    # 6. Row-wise L2 Normalization (makes Euclidean distance monotonic with Cosine distance)
    X_normalized: sparse.csr_matrix = normalize(X_combined, norm="l2", axis=1)

    feature_names = genre_names + kw_names + ov_names + num_names
    print(f"Built feature matrix of shape {X_normalized.shape}")
    return X_normalized, feature_names


def save_to_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH_CLEANED) -> None:
    """Saves the cleaned DataFrame into a local SQLite database with indexes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        filename = "movies_cleaned"
        df.to_sql(filename, conn, if_exists="replace", index=False)
        conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_movies_id ON {filename} (id);")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_movies_vote_average ON {filename} (vote_average DESC);"
        )
        conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_movies_popularity ON {filename} (popularity DESC);"
                )

    print(f"Saved {len(df):,} movies to SQLite database at: {db_path}")


def run_preprocess_pipeline() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_movies_from_sqlite()

    X_weighted, feature_names = build_feature_matrix(df)

    # Save sparse feature matrix and feature vocabulary
    sparse.save_npz(PROCESSED_DIR / "X_weighted.npz", X_weighted)
    with open(PROCESSED_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Save movie index alignment table for downstream clustering
    df[["id", "title", "release_year", "genres", "vote_average", "popularity"]].to_parquet(
        PROCESSED_DIR / "movies_index.parquet", index=False
    )
    df[["id", "title", "release_year", "genres", "vote_average", "popularity"]].to_csv(
            PROCESSED_DIR / "movies_index.csv", index=False
        )

    ## Saving the "movies_index.csv file to movies_cleaned.db
    save_to_sqlite(df[["id", "title", "release_year", "genres", "vote_average", "popularity"]], DB_PATH_CLEANED)

    print(f"Saved preprocessed artifacts to {PROCESSED_DIR}")


if __name__ == "__main__":
    run_preprocess_pipeline()