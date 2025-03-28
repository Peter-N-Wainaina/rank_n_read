import pytest 
import os 

from backend.dataset import Dataset
from test_constants import BOOKS_TEST_JSON


@pytest.fixture
def dataset() -> Dataset:
    """Fixture to create a Dataset instance"""
    return Dataset(BOOKS_TEST_JSON)

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

