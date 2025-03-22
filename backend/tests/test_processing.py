import os
import pytest
from backend.processing import Processor
from test_constants import PROCESSOR_TEST_JSON

@pytest.fixture
def processor() -> Processor:
    """Fixture to create a Processor instance"""
    return Processor(PROCESSOR_TEST_JSON)

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


