"""Central configuration for the movie recommender.

Tweak these values to change filtering thresholds, model behaviour, or the
location of the dataset without touching the rest of the codebase.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "TMDB_movie_dataset_v11.csv"

# --- Dataset ---------------------------------------------------------------
# Only the columns we actually use are loaded (keeps memory usage low).
USE_COLUMNS = [
    "title",
    "runtime",
    "release_date",
    "genres",
    "production_countries",
    "popularity",
    "vote_count",
    "vote_average",
]

# --- Data filtering --------------------------------------------------------
MIN_VOTE_COUNT = 500      # ignore movies with too few votes (noisy ratings)
MIN_VOTE_AVERAGE = 0.0    # ignore unrated entries
TOP_N_GENRES = 10         # keep only the N most common genres
TOP_N_COUNTRIES = 10      # keep only the N most common production countries

# --- Features --------------------------------------------------------------
NUMERIC_FEATURES = ["runtime", "release_year", "popularity", "vote_count"]
CATEGORICAL_FEATURES = ["genres", "production_countries"]

# --- Model / recommendations ----------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_RECOMMENDATIONS = 5
SHORTLIST_MULTIPLIER = 4   # similarity shortlist size = N * this, then re-ranked

# --- Rating bounds ---------------------------------------------------------
MIN_RATING = 1.0
MAX_RATING = 10.0
