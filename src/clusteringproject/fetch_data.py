import sqlite3
from pathlib import Path
import kagglehub
import pandas as pd

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "movies.db"

RELEVANT_COLUMNS = [
    "id",
    "title",
    "release_date",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
    "genres",
    "keywords",
    "overview",
    "original_language",
    "poster_path",
]


def download_and_load_raw_csv() -> pd.DataFrame:
    """Downloads the TMDB dataset from Kaggle and loads only relevant columns."""
    print("Downloading TMDB dataset via kagglehub...")
    dataset_dir = Path(
        kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
    )

    csv_files = list(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {dataset_dir}")

    csv_path = csv_files[0]
    print(f"Loading raw data from {csv_path.name}...")

    df = pd.read_csv(
        csv_path,
        usecols=lambda col: col in RELEVANT_COLUMNS or col in {"status", "adult"},
        low_memory=False,
    )
    return df


def clean_and_filter_movies(df: pd.DataFrame, top_n: int = 10000) -> pd.DataFrame:
    """Filters out adult/unreleased/short films, extracts release_year, and keeps top N by vote_count."""
    print(f"Initial row count: {len(df):,}")

    # 1. Filter out adult films and unreleased titles if columns exist
    if "adult" in df.columns:
        df = df[df["adult"].astype(str).str.lower() == "false"]
    if "status" in df.columns:
        df = df[df["status"] == "Released"]

    # 2. Require valid title, genres, keywords, and minimum vote threshold
    df = df.dropna(subset=["id", "title", "genres", "release_date"])
    df = df[df["vote_count"] >= 50]
    df = df[(df["runtime"] >= 40) & (df["runtime"] <= 300)]

    # 3. Extract numerical release_year
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)

    # 4. Deduplicate by TMDB ID and take the top_n movies by vote_count
    df = df.drop_duplicates(subset=["id"])
    df = df.sort_values(by="vote_count", ascending=False).head(top_n).copy()

    # 5. Clean missing text fields
    df["keywords"] = df["keywords"].fillna("")
    df["overview"] = df["overview"].fillna("")
    df["original_language"] = df["original_language"].fillna("en")
    df["poster_path"] = df["poster_path"].fillna("")

    final_cols = [
        "id",
        "title",
        "release_year",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
        "genres",
        "keywords",
        "overview",
        "original_language",
        "poster_path",
    ]
    df = df[final_cols].reset_index(drop=True)
    print(f"Cleaned dataset down to top {len(df):,} movies.")
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """Saves the cleaned DataFrame into a local SQLite database with indexes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        df.to_sql("movies", conn, if_exists="replace", index=False)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_movies_id ON movies (id);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_movies_votes ON movies (vote_count DESC);"
        )

    print(f"Saved {len(df):,} movies to SQLite database at: {db_path}")


def run_fetch_pipeline(top_n: int = 10000) -> pd.DataFrame:
    raw_df = download_and_load_raw_csv()
    clean_df = clean_and_filter_movies(raw_df, top_n=top_n)
    save_to_sqlite(clean_df)
    return clean_df


if __name__ == "__main__":
    run_fetch_pipeline(top_n=10000)
