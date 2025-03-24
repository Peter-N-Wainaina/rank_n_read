import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from backend.processing import Processor
from test_constants import PROCESSOR_TEST_JSON

@pytest.fixture
def processor() -> Processor:
    """Fixture to create a Processor instance"""
    return Processor(PROCESSOR_TEST_JSON)

@pytest.fixture
def mock_recommendation_sources() -> Processor:
    """Fixture that makes get_recs_by_* functions"""
    processor = Processor()
    processor.books = {
        "book1": [{"title": "Book 1"}],
        "book2": [{"title": "Book 2"}],
        "book3": [{"title": "Book 3"}],
        "book4": [{"title": "Book 4"}],
        "book5": [{"title": "Book 5"}]
    }
    processor.get_recs_from_title = MagicMock(return_value={"book1": 0.3, "book2":0.5, "book4":0.2})
    processor.get_recs_from_author = MagicMock(return_value={"book2": 0.2, "book3":0.2, "book4":0.6})
    processor.get_recs_from_categories = MagicMock(return_value={"book3":0.1,"book5": 0.9})
    return processor

def test_compute_jaccard_similarity(processor):
    set1 = {"fear"}
    set2 = {"fear"}
    sim1 = processor.compute_jaccard_similarity(set1, set2)
    assert sim1 == 1.0, f"Expected 1.0 but got {sim1}"

    set3 = {"fear"}
    set4 = {"fear", "love"}
    sim2 = processor.compute_jaccard_similarity(set3, set4)
    assert sim2 == 0.5, f"Expected 0.5 but got {sim2}"
    
    set5 = {"religion"}
    set6 = {"fear", "love"}
    sim3 = processor.compute_jaccard_similarity(set5, set6)
    assert sim3 == 0.0, f"Expected 0.0 but got {sim3}"


def test_get_recs_from_categories(processor):
    # one category
    recs1 = processor.get_recs_from_categories(["Nonfiction"])
    max_title = max(recs1, key=recs1.get)
    max_value = recs1[max_title]
    assert max_value == 1/3, f"Expected 1/3 but got {max_value}"
    assert max_title == "Sample Book Two", f"Expected Sample Book Two but got {max_title}"

    # multiple categories
    recs2 = processor.get_recs_from_categories(["Science Fiction", "Technology"])
    max_title = max(recs2, key=recs2.get)
    max_value = recs2[max_title]
    assert max_value == 1, f"Expected 1 but got {max_value}"
    assert max_title == "Multi-Author Book", f"Expected 'Multi-Author Book' but got {max_title}"

    # same catefories but some are compound
    recs3 = processor.get_recs_from_categories(["Science Fiction & Technology"])
    max_title = max(recs3, key=recs3.get)
    max_value = recs3[max_title]
    assert max_value == 1, f"Expected 1 but got {max_value}"
    assert max_title == "Multi-Author Book", f"Expected 'Multi-Author Book' but got {max_title}"

def test_get_recs_from_author(processor):
    # partial match
    recs1 = processor.get_recs_from_author(["Rawling"])
    max_title = max(recs1, key=recs1.get)
    assert max_title == "Just science fiction", f"Exepected 'Just science fiction' but got {max_title}"

    # full match
    recs2 = processor.get_recs_from_author(["Charlie Coauthor"])
    max_title = max(recs2, key=recs2.get)
    assert max_title == "Edge Case Book", f"Exepected 'Edge Case Book' but got {max_title}"


    recs3 = processor.get_recs_from_author(["Eli Coauthor"])
    max_title = max(recs3, key=recs3.get)
    assert max_title == "Multi-Author Book", f"Exepected 'Multi-Author Book' but got {max_title}"


MOCK_USER_INPUT = {
        "titles": ["Some Title"],
        "authors": ["Author Name"],
        "categories": ["Category Name"]
    }
def test_recommendations_with_default_weights(mock_recommendation_sources):
    processor = mock_recommendation_sources
    weights = SimpleNamespace(TITLES=1, AUTHORS=1, CATEGORIES=1)
    recs = processor.get_recommended_books(MOCK_USER_INPUT, 3, weights=weights)

    assert len(recs) == 3

    titles = [book["title"] for book in recs]
    assert set(titles) == {"Book 2", "Book 4", "Book 5"}

    scores = [book["score"] for book in recs]
    assert scores == sorted(scores, reverse=True)

def test_recommendations_with_custom_weights(mock_recommendation_sources):
    processor = mock_recommendation_sources

    weights = SimpleNamespace(TITLES=0.3, AUTHORS=0.0, CATEGORIES=0.7)
    recs = processor.get_recommended_books(MOCK_USER_INPUT, 1, weights)

    assert recs[0]["title"] == "Book 5"
    assert len(recs) == 1

def test_recommendations_with_small_output_size(mock_recommendation_sources):
    processor = mock_recommendation_sources
    weights = SimpleNamespace(TITLES=1, AUTHORS=1, CATEGORIES=1)
    recs = processor.get_recommended_books(MOCK_USER_INPUT, 1, weights)

    assert recs[0]["title"] == "Book 5"
