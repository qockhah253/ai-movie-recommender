"""Generate a small, synthetic, TMDB-shaped dataset.

This lets you run the app and the tests without downloading the full ~400MB
Kaggle dataset. The data is **made up** (random titles, ratings, etc.) and is
only meant for trying the app out — replace it with the real
``TMDB_movie_dataset_v11.csv`` for meaningful results.

Usage:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from movie_recommender import config  # noqa: E402

_ADJECTIVES = [
    "Crimson", "Silent", "Last", "Broken", "Golden", "Eternal", "Hidden",
    "Savage", "Distant", "Frozen", "Burning", "Lost", "Velvet", "Iron",
    "Midnight", "Scarlet", "Wild", "Quiet", "Falling", "Endless",
]
_NOUNS = [
    "Horizon", "Echo", "Empire", "Voyage", "Shadow", "Promise", "Kingdom",
    "Mirage", "Requiem", "Harbor", "Legacy", "Storm", "Garden", "Verdict",
    "Odyssey", "Tide", "Dream", "Crown", "River", "Signal",
]
_GENRES = [
    "Drama", "Comedy", "Action", "Thriller", "Horror", "Romance", "Adventure",
    "Animation", "Crime", "Science Fiction", "Fantasy", "Mystery",
]
_COUNTRIES = [
    "United States of America", "United Kingdom", "France", "Japan",
    "South Korea", "Germany", "India", "Canada", "Spain", "Italy",
]


def generate(path: Path | str, n_rows: int = 1200, seed: int = 7) -> Path:
    """Write a synthetic TMDB-shaped CSV to ``path`` and return the path."""
    rng = np.random.default_rng(seed)

    def title() -> str:
        return f"{rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)}"

    def multi(options) -> str:
        # Comma-separated, exactly like the real TMDB dataset.
        k = int(rng.integers(1, 4))
        return ", ".join(rng.choice(options, size=k, replace=False))

    df = pd.DataFrame(
        {
            "title": [title() for _ in range(n_rows)],
            "runtime": rng.integers(80, 175, n_rows),
            "release_date": pd.to_datetime(
                rng.integers(1990, 2024, n_rows).astype(str) + "-01-01"
            ),
            "genres": [multi(_GENRES) for _ in range(n_rows)],
            "production_countries": [multi(_COUNTRIES) for _ in range(n_rows)],
            "popularity": rng.uniform(5, 180, n_rows).round(2),
            "vote_count": rng.integers(120, 9000, n_rows),
            "vote_average": rng.uniform(4.5, 8.9, n_rows).round(1),
        }
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    out = generate(config.DATASET_PATH)
    print(f"Sample dataset written to: {out}")
    print("NOTE: this is synthetic data for testing only — replace it with the "
          "real TMDB_movie_dataset_v11.csv for real results.")


if __name__ == "__main__":
    main()
