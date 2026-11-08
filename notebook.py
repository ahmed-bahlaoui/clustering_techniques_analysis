import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import kagglehub
    import polars as pl
    from pathlib import Path

    return kagglehub, pl


@app.cell
def _(kagglehub, pl):

    # Download latest version
    path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")

    # Log dataset path to terminal
    print("Path to dataset files:", path)

    datasetname = "TMDB_movie_dataset_v11.csv"
    df = pl.read_csv(f"{path}/{datasetname}")
    return


@app.cell
def _():
    ## Data Preprocessing
    # Viewing the data

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
