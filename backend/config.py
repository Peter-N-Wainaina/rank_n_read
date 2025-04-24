import os

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

BASE_DATA_PATH = os.path.join(ROOT_PATH,'data')
POPULAR_BOOKS_JSON_FILE = os.path.join(BASE_DATA_PATH,'popular_books.json')
SAMPLED_BOOKS_JSON_FILE = os.path.join(BASE_DATA_PATH,'sampled_books.json')
DEFAULT_REVIEWS_JSON_FILE  = os.path.join(BASE_DATA_PATH,'sampled_reviews.json')