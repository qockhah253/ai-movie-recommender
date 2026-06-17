"""Dataset loading, cleaning, and feature encoding."""

from __future__ import annotations

import pandas as pd

from . import config


class MovieData:
    """Load, clean, and one-hot encode the TMDB movie dataset.

    Attributes:
        df: The cleaned movies (one row per movie).
        features: One-hot encoded feature matrix aligned with ``df``.
        feature_columns: Ordered list of feature column names.
    """

    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path or config.DATASET_PATH
        self.df = None
        self.features = None
        self.feature_columns = None

    def load(self) -> "MovieData":
        """Read the CSV, preprocess it, and build the feature matrix."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.dataset_path}'.\n"
                "Download 'TMDB_movie_dataset_v11.csv' from Kaggle "
                "(asaniczka/tmdb-movies-dataset-2023-930k-movies) and place "
                f"it in '{config.DATA_DIR}'."
            )

        df = pd.read_csv(self.dataset_path, usecols=config.USE_COLUMNS)
        self.df = self._preprocess(df)
        self._build_features()
        return self

    @staticmethod
    def _first_value(value) -> str:
        """Return the primary value of a comma-separated TMDB field.

        TMDB v11 stores multi-valued fields as comma-separated strings, e.g.
        ``"Action, Adventure, Fantasy"`` or ``"United States of America"``.
        """
        return str(value).split(",")[0].strip()

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()

        # Quality filters: enough votes and an actual rating.
        df = df[df["vote_count"] > config.MIN_VOTE_COUNT]
        df = df[df["vote_average"] > config.MIN_VOTE_AVERAGE]

        # Derive release year from the release date.
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df = df.dropna(subset=["release_date"])
        df["release_year"] = df["release_date"].dt.year

        # Keep only the primary genre and country.
        df["genres"] = df["genres"].apply(self._first_value)
        df["production_countries"] = df["production_countries"].apply(self._first_value)

        # Restrict to the most common categories to avoid sparse dummies.
        top_genres = df["genres"].value_counts().nlargest(config.TOP_N_GENRES).index
        top_countries = (
            df["production_countries"]
            .value_counts()
            .nlargest(config.TOP_N_COUNTRIES)
            .index
        )
        df = df[
            df["genres"].isin(top_genres)
            & df["production_countries"].isin(top_countries)
        ]

        return df.reset_index(drop=True)

    def _build_features(self) -> None:
        encoded = pd.get_dummies(
            self.df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES],
            columns=config.CATEGORICAL_FEATURES,
        )
        self.features = encoded
        self.feature_columns = list(encoded.columns)

    @property
    def genres(self):
        """Sorted list of unique genres available in the cleaned data."""
        return sorted(self.df["genres"].unique())

    @property
    def countries(self):
        """Sorted list of unique production countries in the cleaned data."""
        return sorted(self.df["production_countries"].unique())
