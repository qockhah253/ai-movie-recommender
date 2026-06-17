"""Smoke tests for the data + model pipeline (no GUI required)."""

from movie_recommender import config
from movie_recommender.data import MovieData
from movie_recommender.model import RecommenderModel
from scripts.generate_sample_data import generate


def _make_data(tmp_path) -> MovieData:
    csv = tmp_path / "movies.csv"
    generate(csv, n_rows=800, seed=1)
    return MovieData(dataset_path=csv).load()


def test_primary_value_split():
    # The key bug fix: comma-separated TMDB fields must be split on commas.
    assert MovieData._first_value("Action, Adventure, Fantasy") == "Action"
    assert (
        MovieData._first_value("United States of America")
        == "United States of America"
    )


def test_data_loads_and_cleans(tmp_path):
    data = _make_data(tmp_path)
    assert len(data.df) > 0
    assert len(data.genres) <= config.TOP_N_GENRES
    assert len(data.countries) <= config.TOP_N_COUNTRIES
    for column in config.NUMERIC_FEATURES:
        assert column in data.feature_columns


def test_model_trains_and_compares(tmp_path):
    data = _make_data(tmp_path)
    rec = RecommenderModel(data).train()
    assert rec.best_model is not None
    assert len(rec.scores) == 3  # three candidate models compared
    assert rec.best_name in {m["model"] for m in rec.scores}


def test_predicted_rating_in_range(tmp_path):
    data = _make_data(tmp_path)
    rec = RecommenderModel(data).train()
    rating = rec.predict_rating(120, 2020, data.genres[0], data.countries[0])
    assert config.MIN_RATING <= rating <= config.MAX_RATING


def test_recommendations_shape(tmp_path):
    data = _make_data(tmp_path)
    rec = RecommenderModel(data).train()
    recs = rec.recommend(120, 2020, data.genres[0], data.countries[0], n=5)
    assert len(recs) == 5
    assert "title" in recs.columns
