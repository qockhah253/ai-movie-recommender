"""Model training with automatic model comparison and recommendations.

Several regression models are trained and evaluated on a held-out test set;
the best one (highest R^2) is used to predict ratings. Recommendations are
generated with cosine similarity over the scaled feature space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from . import config
from .data import MovieData


def build_candidates(random_state: int) -> dict:
    """Return the set of regression models to compare.

    Linear Regression is kept as an interpretable baseline; the tree-based
    models usually capture the non-linear patterns better.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=random_state),
    }


class RecommenderModel:
    """Train and compare models, then recommend similar, well-rated movies."""

    def __init__(self, data: MovieData):
        self.data = data
        self.scaler = MinMaxScaler()
        self.models: dict = {}        # name -> fitted estimator
        self.scores: list = []        # list of metric dicts (one per model)
        self.best_name = None
        self.best_model = None
        self.features_scaled = None

    def train(self) -> "RecommenderModel":
        """Fit the scaler, train every candidate, and select the best model."""
        X = self.data.features.values.astype(float)
        y = self.data.df["vote_average"].values.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )

        # Fit the scaler on the training split only (avoid data leakage).
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        for name, model in build_candidates(config.RANDOM_STATE).items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            self.models[name] = model
            self.scores.append(
                {
                    "model": name,
                    "r2": r2_score(y_test, y_pred),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                }
            )

        best = max(self.scores, key=lambda s: s["r2"])
        self.best_name = best["model"]
        self.best_model = self.models[self.best_name]

        # Pre-scale every movie once for fast similarity search.
        self.features_scaled = self.scaler.transform(X)
        return self

    @property
    def best_metrics(self) -> dict:
        """Metrics of the selected best model."""
        return next(s for s in self.scores if s["model"] == self.best_name)

    def comparison_table(self) -> pd.DataFrame:
        """Model comparison sorted from best to worst R^2."""
        return (
            pd.DataFrame(self.scores)
            .sort_values("r2", ascending=False)
            .reset_index(drop=True)
        )

    def _build_input_vector(self, runtime, release_year, genre, country) -> np.ndarray:
        """Turn user inputs into a feature vector matching the training columns."""
        row = {col: 0.0 for col in self.data.feature_columns}
        row["runtime"] = float(runtime)
        row["release_year"] = float(release_year)
        # Use medians for fields the user doesn't control, to reduce noise.
        row["popularity"] = float(self.data.df["popularity"].median())
        row["vote_count"] = float(self.data.df["vote_count"].median())

        genre_col = f"genres_{genre}"
        country_col = f"production_countries_{country}"
        if genre_col in row:
            row[genre_col] = 1.0
        if country_col in row:
            row[country_col] = 1.0

        vector = [row[col] for col in self.data.feature_columns]
        return np.array(vector, dtype=float).reshape(1, -1)

    def predict_rating(self, runtime, release_year, genre, country) -> float:
        """Predict the expected vote average using the best model."""
        scaled = self.scaler.transform(
            self._build_input_vector(runtime, release_year, genre, country)
        )
        rating = float(self.best_model.predict(scaled)[0])
        return float(np.clip(rating, config.MIN_RATING, config.MAX_RATING))

    def recommend(self, runtime, release_year, genre, country, n=None) -> pd.DataFrame:
        """Return the top-N similar movies, re-ranked by actual rating."""
        n = n or config.N_RECOMMENDATIONS
        scaled = self.scaler.transform(
            self._build_input_vector(runtime, release_year, genre, country)
        )

        similarity = cosine_similarity(scaled, self.features_scaled)[0]

        ranked = self.data.df.copy()
        ranked["similarity"] = similarity
        shortlist = ranked.nlargest(n * config.SHORTLIST_MULTIPLIER, "similarity")
        return shortlist.sort_values(
            ["vote_average", "similarity"], ascending=False
        ).head(n)
