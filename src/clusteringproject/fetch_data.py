import kagglehub
import polars as pl
from pathlib import Path

# Download latest version
output_dir = Path.cwd() / "data" / "raw"
path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies", output_dir=str(output_dir))

# Log dataset path to terminal 
print("Path to dataset files:", path)
datasetname = "TMDB_movie_dataset_v11.csv"

print(f"Dataset fetched to from Kaggle\npath: {path}\\{datasetname}")


