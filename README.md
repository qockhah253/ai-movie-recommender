# 🎬 AI Movie Recommender

A desktop application that **predicts a movie's rating** and **recommends similar, well-rated movies** from simple attributes (runtime, release year, genre, country). It trains and compares several regression models — **Linear Regression**, **Random Forest**, and **Gradient Boosting** — and automatically uses the best one. Built with **scikit-learn** and a **CustomTkinter** GUI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![CI](https://github.com/qockhah253/ai-movie-recommender/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Features

- **Model comparison** — trains Linear Regression (baseline), Random Forest, and Gradient Boosting, evaluates each on a held-out test set (R², MAE, RMSE), and automatically selects the best.
- **Rating prediction** — estimates a movie's expected TMDB score (1–10).
- **Smart recommendations** — finds the most similar movies with **cosine similarity**, then re-ranks them by actual rating so suggestions are both relevant *and* good.
- **Runs out-of-the-box** — a sample-data generator lets you try the app without downloading the full dataset.
- **Tested** — a `pytest` suite and GitHub Actions CI keep the pipeline working.

---

## 🖼️ Demo

> _Add a screenshot of the running app to `docs/screenshot.png` and it will appear here._

<!-- ![App screenshot](docs/screenshot.png) -->

---

## 🧠 How It Works

1. **Data preparation** — loads the TMDB dataset, keeps movies with more than 500 votes, extracts the *primary* genre and country, derives the release year, and one-hot encodes the categorical fields.
2. **Model training & comparison** — features are scaled with `MinMaxScaler` (fitted on the training split only). Several models are trained and scored on a held-out test set; the one with the highest R² is selected to predict `vote_average`.
3. **Recommendations** — the chosen attributes are turned into the same feature vector and compared against every movie with cosine similarity. A shortlist of the closest matches is then sorted by rating to produce the final five suggestions.

---

## 📁 Project Structure

```
movie-recommender-linear-regression/
├── main.py                     # Entry point: load data, compare models, launch GUI
├── conftest.py                 # Makes the project importable in tests
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Test dependencies
├── README.md  ·  LICENSE  ·  .gitignore
├── .github/workflows/ci.yml    # GitHub Actions CI
├── data/                       # Dataset goes here (git-ignored)
├── scripts/
│   └── generate_sample_data.py # Create a small synthetic dataset
├── tests/
│   └── test_pipeline.py        # Unit tests (no GUI needed)
└── movie_recommender/
    ├── config.py               # All tunable constants and paths
    ├── data.py                 # Loading, cleaning, feature encoding
    ├── model.py                # Training, comparison, recommendations
    └── gui.py                  # CustomTkinter interface
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer

### 1. Clone and install

```bash
git clone https://github.com/qockhah253/ai-movie-recommender.git
cd movie-recommender-linear-regression

# (recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 2a. Quickstart — try it with sample data (no download)

```bash
python scripts/generate_sample_data.py   # writes a synthetic data/ file
python main.py
```

> The sample data is **synthetic** (random titles and ratings) and only for trying the app out.

### 2b. Run with the real dataset

Download the **TMDB Movies Dataset** from Kaggle —
[asaniczka/tmdb-movies-dataset-2023-930k-movies](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) —
and place `TMDB_movie_dataset_v11.csv` in a `data/` folder at the project root:

```
data/TMDB_movie_dataset_v11.csv
```

Then run:

```bash
python main.py
```

On startup the console prints the model comparison, for example:

```
Model comparison (held-out test set):
            model    r2   mae  rmse
Gradient Boosting 0.18x 0.6xx 0.8xx
    Random Forest 0.15x 0.6xx 0.8xx
Linear Regression 0.09x 0.7xx 0.9xx

Best model: Gradient Boosting
```

*(Exact numbers depend on the dataset; ratings are only weakly predictable from these features, so values are modest by design.)*

---

## 🧪 Running Tests

```bash
pip install -r requirements-dev.txt
python scripts/generate_sample_data.py   # tests also generate their own data
pytest
```

---

## ⚙️ Configuration

All tunable values live in `movie_recommender/config.py` — minimum vote count, number of genres/countries to keep, train/test split ratio, number of recommendations, and the dataset path. Change them there without touching the rest of the code.

---

## ⚠️ Notes & Limitations

- A movie's rating is only weakly linear in attributes like runtime and genre, so R² is modest by design — this project is a clear, end-to-end demonstration of regression + similarity, not a state-of-the-art recommender.
- Only the *primary* genre and country are used to keep the feature space small.
- Recommendations are content-based (attribute similarity), not collaborative — there is no per-user history.

---

## 🛣️ Possible Improvements

- Use all genres via multi-hot encoding instead of just the first.
- Add semantic similarity on the movie overview text (e.g. Sentence-BERT embeddings).
- Switch to an interaction dataset (e.g. MovieLens) for collaborative filtering.
- Cache the trained model so the app starts instantly.

---

## 📝 License

Released under the [MIT License](LICENSE).
