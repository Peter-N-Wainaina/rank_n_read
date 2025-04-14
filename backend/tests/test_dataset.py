import pytest 
import os 

from backend.dataset import Dataset
from test_constants import BOOKS_TEST_JSON, REVIEWS_TEST_JSON

@pytest.fixture
def dataset() -> Dataset:
    """Fixture to create a Dataset instance"""
    return Dataset(BOOKS_TEST_JSON, REVIEWS_TEST_JSON)

def test_dataset_init(dataset):
    books = dataset.books    
    assert isinstance(books, dict)
    assert len(books.items()) == 4

    assert dataset.num_books == 4

    categ_index = dataset.categories_index
    categ_index_keys = categ_index.keys()
    assert len(categ_index_keys) == 6
    assert "nonfiction" in categ_index_keys

    authors_index = dataset.authors_index
    authors_index_keys = authors_index.keys()
    assert len(authors_index_keys) == 8
    assert "o'connor" in authors_index_keys
    assert "j.k" in authors_index_keys
    assert len(authors_index["coauthor"]) == 2  

def test_get_books_by_category(dataset):
    tech_books = dataset.get_books_by_category("Technology")
    assert len(tech_books) == 2
    assert "Sample Book Two" in tech_books

    science_books = dataset.get_books_by_category("ScIence")
    assert len(science_books) == 1
    assert "Multi-Author Book" in tech_books

    assert dataset.get_books_by_category("Astrology") == []
    assert dataset.get_books_by_category(" ") == []

def test_get_books_by_author(dataset):
    coauthor_books = dataset.get_books_by_author("coauthor")
    assert len(coauthor_books) == 2
    assert set(dataset.get_books_by_author("coauthor")) == {
        "Multi-Author Book", "Edge Case Book"
    }

    assert dataset.get_books_by_author("Unknown Author") == []
    assert dataset.get_books_by_author("") == []

@pytest.mark.parametrize("token, expected_title ,expected_length",[
    ("sample","Sample Book Two", 2), 
    ("book", "Multi-Author Book", 4),
    (" ", "", 0)
])
def test_get_books_by_title_token(dataset, token, expected_title, expected_length):
    titles_with_token = dataset.get_books_by_title_token(token)
    assert len(titles_with_token) == expected_length
    if expected_title:
        assert expected_title in titles_with_token

def test_build_title_vocab_frequency():
    dataset = Dataset()
    dataset.books = {
        "The Great Adventure": [{}],
        "Great Expectations": [{}],
        "The Silent Patient": [{}],
    }

    expected = {
        "the": 2,
        "great": 2,
        "adventure": 1,
        "expectations": 1,
        "silent": 1,
        "patient": 1,
    }

    vocab = dataset._build_title_vocab_frequency()
    assert vocab == expected

def test_build_book_data_dict(dataset):
    book_data_dict = dataset.book_data_dict 

    assert isinstance(book_data_dict, dict)
    assert len(book_data_dict) == 4

    data_one = book_data_dict["Sample Book One"]
    assert "test description for book one" in data_one
    assert "j.k rawling" in data_one
    assert "fiction" in data_one
    assert "adventure" in data_one

    data_two = book_data_dict["Edge Case Book"]
    assert "this book has no categories listed" in data_two
    assert "charlie coauthor" in data_two
    assert "edge case book" in data_two

    assert data_one == data_one.lower()
    assert data_two == data_two.lower()

def test_merged_reviews_exist(dataset):
    book = dataset.books["Sample Book One"][0]
    assert "avg_rating" in book
    assert "price" in book
    assert "reviews" in book

    assert isinstance(book["avg_rating"], float) or book["avg_rating"] is None
    assert isinstance(book["price"], str) or book["price"] is None
    assert isinstance(book["reviews"], list)

def test_reviews_preserved_correctly(dataset):
    book = dataset.books["Multi-Author Book"][0]
    assert isinstance(book["reviews"], list)
    assert len(book["reviews"]) == 3
    assert "collaboration" in book["reviews"][1].lower()

def test_books_with_missing_reviews(dataset):
    book = dataset.books["Sample Book Two"][0]
    assert book["reviews"] == []
    assert book["avg_rating"] is None
    assert book["price"] is None
