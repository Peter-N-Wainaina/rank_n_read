import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from scipy.sparse import csr_matrix

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



# Test for compute_tf method
def test_compute_tf(processor):
    tokens = ['catcher', 'rye', 'in', 'the', 'catcher']
    
    # Test Term Frequency calculation
    tf = processor.compute_tf(tokens)
    
    # Expected term frequencies for the tokens
    expected_tf = np.array([2/5, 1/5, 1/5, 1/5])  # 'catcher' appears twice, others appear once
    
    np.testing.assert_array_almost_equal(tf, expected_tf, err_msg="TF calculation failed!")


# Test for compute_idf method
def test_compute_idf(processor):
    inverted_index = {
        'catcher': ['Book1', 'Book2'],
        'rye': ['Book1', 'Book3'],
        'in': ['Book1', 'Book4'],
        'the': ['Book2', 'Book4']
    }
    
    total_relevant_documents = 4
    idf_catcher = processor.compute_idf('catcher', inverted_index, total_relevant_documents)
    idf_rye = processor.compute_idf('rye', inverted_index, total_relevant_documents)
    
    expected_idf_catcher = np.log(4 / 2)  
    expected_idf_rye = np.log(4 / 2)

    assert np.isclose(idf_catcher, expected_idf_catcher), f"Expected IDF for 'catcher' is {expected_idf_catcher}, but got {idf_catcher}"
    assert np.isclose(idf_rye, expected_idf_rye), f"Expected IDF for 'rye' is {expected_idf_rye}, but got {idf_rye}"


# Test for compute_tfidf method
def test_compute_tfidf(processor):
    tokens = ['catcher', 'rye', 'in', 'the']
    inverted_index = {
        'catcher': ['Book1', 'Book2'],
        'rye': ['Book1', 'Book3'],
        'in': ['Book1', 'Book4'],
        'the': ['Book2', 'Book4']
    }
    total_relevant_documents = 4
    vocab = ['catcher', 'rye', 'in', 'the']
    
    tfidf = processor.compute_tfidf(tokens, inverted_index, total_relevant_documents, vocab)
    
    # Manually compute expected TF-IDF for each term
    tf_catcher = 1 / 4  
    tf_rye = 1 / 4    
    tf_in = 1 / 4  
    tf_the = 1 / 4  

    idf_catcher = np.log(4 / 2) 
    idf_rye = np.log(4 / 2)    
    idf_in = np.log(4 / 2) 
    idf_the = np.log(4 / 2) 

    # Expected TF-IDF values for each term in vocab
    expected_tfidf = np.array([
        tf_catcher * idf_catcher, 
        tf_rye * idf_rye, 
        tf_in * idf_in,  
        tf_the * idf_the 
    ])
    
    # Check if computed TF-IDF matches expected values
    np.testing.assert_array_almost_equal(tfidf, expected_tfidf, err_msg="TF-IDF calculation failed!")


# Test for cosine_similarity method
def test_cosine_similarity(processor):
    query_tfidf = np.array([1, 0]) 
    book_tfidf = np.array([1, 0])   
    similarity = processor.cosine_similarity(query_tfidf, book_tfidf)
    assert similarity == 1.0, f"Expected 1.0 but got {similarity}"

    book_tfidf_orthogonal = np.array([0, 1]) 
    similarity_orthogonal = processor.cosine_similarity(query_tfidf, book_tfidf_orthogonal)
    assert similarity_orthogonal == 0.0, f"Expected 0.0 but got {similarity_orthogonal}"



# Test for create_inverted_index_for_title method
def test_create_inverted_index_for_title(processor):
    processor.dataset.get_books_by_title_token = MagicMock(return_value=["Book1", "Book2"])

    title = "The Catcher in the Rye"
    inverted_index, relevant_docs, total_relevant_documents = processor.create_inverted_index_for_title(title)

    expected_inverted_index = {
        'catcher': ['Book1', 'Book2'],
        'rye': ['Book1', 'Book2'],
        'the': ['Book1', 'Book2'],
        'in': ['Book1', 'Book2']
    }
    
    assert inverted_index == expected_inverted_index, f"Expected {expected_inverted_index} but got {inverted_index}"
    assert relevant_docs == {'Book1', 'Book2'}, f"Expected relevant docs to be {{'Book1', 'Book2'}} but got {relevant_docs}"
    assert total_relevant_documents == 2, f"Expected total relevant documents to be 2 but got {total_relevant_documents}"


# Test for get_recs_from_title method
def test_get_recs_from_title(processor):
    processor.dataset.get_books_by_title_token = MagicMock(return_value=["Book1", "Book2", "Book3"])
    
    title = "The Catcher in the Rye"
    recommended_books = processor.get_recs_from_title(title, top_n=3)
    
    assert isinstance(recommended_books, dict), "Expected the result to be a dictionary"
    assert len(recommended_books) == 3, f"Expected 3 recommendations, but got {len(recommended_books)}"
    assert "Book1" in recommended_books, "Expected 'Book1' to be in the recommendations"

MOCK_USER_INPUT_NO_DESCRIPTION = {
        "titles": ["Some Title"],
        "authors": ["Author Name"],
        "categories": ["Category Name"],
        "description": ""
    }
def test_recommendations_with_default_weights(mock_recommendation_sources):
    processor = mock_recommendation_sources
    weights = SimpleNamespace(TITLES=1, AUTHORS=1, CATEGORIES=1)
    recs = processor.get_recommended_books(MOCK_USER_INPUT_NO_DESCRIPTION, 3, weights=weights)

    assert len(recs) == 3

    titles = [book["title"] for book in recs]
    assert set(titles) == {"Book 2", "Book 4", "Book 5"}

    scores = [book["score"] for book in recs]
    assert scores == sorted(scores, reverse=True)

def test_recommendations_with_custom_weights(mock_recommendation_sources):
    processor = mock_recommendation_sources

    weights = SimpleNamespace(TITLES=0.3, AUTHORS=0.0, CATEGORIES=0.7)
    recs = processor.get_recommended_books(MOCK_USER_INPUT_NO_DESCRIPTION, 1, weights)

    assert recs[0]["title"] == "Book 5"
    assert len(recs) == 1

def test_recommendations_with_small_output_size(mock_recommendation_sources):
    processor = mock_recommendation_sources
    weights = SimpleNamespace(TITLES=1, AUTHORS=1, CATEGORIES=1)
    recs = processor.get_recommended_books(MOCK_USER_INPUT_NO_DESCRIPTION, 1, weights)

    assert recs[0]["title"] == "Book 5"

def test_remove_common_words_removes_expected_tokens():
    processor = Processor()
    processor.title_vocab = {
        "the": 5,      # 100%
        "of": 4,       # 80%
        "great": 2,    # 40%
        "escape": 1    # 20%
    }
    num_books = 5
    result = processor.remove_common_words_from_title("The Great Escape",num_books, frequency_threshold=0.5)
    assert result == "great escape"

def test_remove_common_words_removes_more_with_lower_threshold():
    processor = Processor()
    processor.title_vocab = {
        "the": 5,
        "great": 4,
        "escape": 1
    }
    num_books = 5
    result = processor.remove_common_words_from_title("The Great Escape",num_books,frequency_threshold=0.3)
    assert result == "escape"

def test_remove_common_words_with_high_threshold_removes_nothing():
    processor = Processor()
    processor.title_vocab = {
        "the": 5,
        "great": 4,
        "escape": 1
    }
    num_books = 6
    result = processor.remove_common_words_from_title("The Great Escape", num_books, frequency_threshold=1)
    assert result == "the great escape"

def test_remove_common_words_all_tokens_removed():
    processor = Processor()
    processor.title_vocab = {
        "the": 5,
        "great": 5,
        "escape": 5
    }
    num_books = 5
    result = processor.remove_common_words_from_title("The Great Escape", num_books,frequency_threshold=0.8)
    assert result == ""

def test_create_tfidf_matrix():
    processor = Processor(PROCESSOR_TEST_JSON)
    titles, tfidf_matrix, vectorizer = processor.create_tfidf_matrix()
    books = processor.books

    # Assertions
    assert isinstance(titles, list), "Titles should be a list"
    assert isinstance(tfidf_matrix, csr_matrix), "TF-IDF matrix should be a sparse matrix"
    assert isinstance(vectorizer, TfidfVectorizer), "Should return a TfidfVectorizer"

    assert len(titles) == len(books), "Number of titles should match number of books"
    assert tfidf_matrix.shape[0] == len(books), "Number of rows in TF-IDF matrix should match number of books"


def test_transform_query():
    processor = Processor()
    books = {
        "Book One": "A thrilling journey through space and time.",
        "Book Two": "A deep dive into technological advancements and AI.",
        "Book Three": "Romance and mystery woven in a historical setting."
    }
    descriptions = list(books.values())

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    svd = TruncatedSVD(n_components=2, random_state=42)
    svd.fit(tfidf_matrix)

    query_text = "space and AI"
    reduced_query = processor.transform_query(query_text, vectorizer, svd)

    assert isinstance(reduced_query, np.ndarray), "Output should be a numpy array"
    assert reduced_query.shape == (1, 2), f"Expected shape (1, 2) but got {reduced_query.shape}"
    assert not np.isnan(reduced_query).any(), "Output contains NaN values"

#------SVD Tests------#
@pytest.fixture
def mock_svd_setup() -> Processor:
    with patch.object(Processor, 'create_tfidf_matrix') as mock_create_tfidf, \
         patch.object(Processor, 'reduce_with_svd') as mock_reduce_svd:
        
        # Define mock returns
        mock_create_tfidf.return_value = (
            ["Book A", "Book B", "Book C"],
            csr_matrix(np.array([[0.1, 0.2], [0.2, 0.1], [0.3, 0.4]])),
            MagicMock()  # vectorizer
        )

        mock_reduce_svd.return_value = (
            np.array([[0.2, 0.8], [0.1, 0.9], [0.5, 0.5]]),
            MagicMock()  # SVD model
        )

        # Now instantiation uses mocks
        processor = Processor()

        # Patch remaining methods directly on instance
        processor.transform_query = MagicMock(return_value=np.array([[0.15, 0.85]]))
        processor.get_top_k_similar_books = MagicMock(return_value={
            "Book B": 0.91,
            "Book A": 0.83
        })

        return processor
