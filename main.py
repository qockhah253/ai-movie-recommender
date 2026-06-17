"""Entry point: load data, compare models, and launch the GUI.

Run with:  python main.py

If you don't have the dataset yet, generate a small synthetic one first:
    python scripts/generate_sample_data.py
"""

from movie_recommender.data import MovieData
from movie_recommender.gui import MovieApp
from movie_recommender.model import RecommenderModel


def main() -> None:
    print("Loading dataset...")
    data = MovieData().load()
    print(f"Loaded {len(data.df):,} movies after cleaning.\n")

    print("Training and comparing models...")
    recommender = RecommenderModel(data).train()

    print("\nModel comparison (held-out test set):")
    table = recommender.comparison_table()
    print(
        table.to_string(
            index=False,
            formatters={
                "r2": "{:.3f}".format,
                "mae": "{:.3f}".format,
                "rmse": "{:.3f}".format,
            },
        )
    )
    print(f"\nBest model: {recommender.best_name}\n")

    MovieApp(recommender).mainloop()


if __name__ == "__main__":
    main()
