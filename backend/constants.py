from types import SimpleNamespace

# Database Book Record Keys
CATEGORY_KEY = "categories"
AUTHOR_KEY = "authors"
SCORE_KEY = "score"
DESCRIPTION_KEY = "description"
AVG_RATING_KEY = "avg_rating"
PRICE_KEY = "price"
REVIEWS_KEY = "reviews"

NOT_AVAILABLE = "NOT AVAILABLE"

# Frontend Input Keys
INPUT_TITLES_KEY = "titles"
INPUT_AUTHORS_KEY = "authors"
INPUT_CATEGORIES_KEY = "categories"
INPUT_DESCRIPTION_KEY = "description"

DEFAULT_RECS_WEIGHTS = SimpleNamespace(
    TITLES =  0.6, 
    AUTHORS = 0.3, 
    CATEGORIES =  0.1
)

DEFAULT_RECS_SIZE = 20

# Processor Constants
NUM_LATENT_SEMANTIC_CONCEPTS = 100