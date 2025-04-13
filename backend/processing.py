import math
import numpy as np
from collections import Counter

from .dataset import Dataset
from .utils import tokenize_text, tokenize_name_list, tokenize_list
from .constants import DEFAULT_RECS_WEIGHTS, SCORE_KEY, INPUT_AUTHORS_KEY,\
    INPUT_CATEGORIES_KEY, INPUT_TITLES_KEY, DEFAULT_RECS_SIZE, NUM_LATENT_SEMANTIC_CONCEPTS

from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class Processor(object):
    def __init__(self, json_file=None):
        dataset = Dataset()
        if json_file is not None:
            dataset = Dataset(json_file)
        self.dataset = dataset
        self.books = self.dataset.books
        self.title_vocab = self.dataset.title_vocab_frequency
        self.book_titles, self.tfidf_matrix, self.vectorizer = self.create_tfidf_matrix(self.books)
        self.book_vecs, self.svd_model = self.reduce_with_svd(self.tfidf_matrix)
    
    def compute_jaccard_similarity(self, query_categories, book_categories):
        """
        Returns the similarity score between `query_categories` and `book_categories`
        according to the Jaccard Similarity Measure

        Parameters:
            query_categories: set
                A set of categories given by the client
            book_categories: set
                A set of categories for the book we're considering for similarity
        
        Returns:
            int:
                The similarity score
        """
        intersection_set = query_categories & book_categories
        union = query_categories | book_categories
        return len(intersection_set) / len(union) if len(union) != 0 else 0
    
    def get_recs_from_categories(self, query: list):
        """
        Returns a list of book recommendations based on the query

        Parameters:
            query : list
                The input qeury list of categories
        Returns:
            dict:
                A dictionary of book titles and their scores based on categrories
        """
        modified_query = tokenize_list(query)
        # put the titles you get from Dataset.get_books_by_category in a set to take care of dups
        titles_by_category_set = set()
        for category in modified_query:
            books_by_cat = self.dataset.get_books_by_category(category)
            if books_by_cat:
                titles_by_category_set.update(books_by_cat)

        # get the categories for each retrived subset and compute jaccard similarity
        # you should have a dict in the form {score: book infor dict...} after this block
        score_title_dict = {}
        query_categories_set = set(modified_query)
        for title in titles_by_category_set:
            book_info = self.books[title][0]
            book_categories_ls = tokenize_list(book_info["categories"])
            book_categories_set = set(book_categories_ls)
            sim_score = self.compute_jaccard_similarity(query_categories_set, book_categories_set)
            score_title_dict[title] = sim_score

        return score_title_dict

    def get_recs_from_author(self, authors: list):
        """
        Returns a list of recommended books based on the author
        Supports both exact matches (e.g., "J.K. Rowling") and partial matches (e.g., "Rowling").

        Parameters:
            author: str
                This is the input query. We are doing one author for now. We might
                add more authors later

        Returns:
            dict:
                A dictionary of book titles and their scores based on their authors
        """
        modified_authors = tokenize_name_list(authors)
        # put the titles you get from Dataset.get_books_by_author in a set to take care of dups
        titles_by_authors_set = set()
        for author in modified_authors:
            books_titles_by_author = self.dataset.get_books_by_author(author)
            if books_titles_by_author:
                titles_by_authors_set.update(books_titles_by_author)
        
        # make a set with the authors in it to pass into sim measure func
        authors_set = set(modified_authors)

        # get the authors for each retrived subset and compute jaccard similarity btwn authors
        # you should have a dict in the form {score: book infor dict...} after this block
        score_books_dict = {}
        for title in titles_by_authors_set:
            book_info = self.books[title][0]
            modified_authors_from_book = tokenize_name_list(book_info["authors"])
            book_authors_set = set(modified_authors_from_book)
            sim_score = self.compute_jaccard_similarity(authors_set, book_authors_set)
            score_books_dict[title] = sim_score
        
        return score_books_dict

    def compute_tf(self, tokens):
        """
        Compute the term frequency (TF) for each token in a list of tokens.
        
        Args:
        tokens (list): A list of tokenized words from a document or query.
        
        Returns:
        np.array: A NumPy array with terms as indices and their respective term frequency (TF) as values.
        """
        term_count = Counter(tokens)
        total_terms = len(tokens)
        tf = np.zeros(len(term_count)) 
        
        for i, (term, count) in enumerate(term_count.items()):
            tf[i] = count / total_terms  
        return tf

    def compute_idf(self, term, inverted_index, total_relevant_documents):
        """
        Compute the inverse document frequency (IDF) of a term using the number of relevant documents.
        
        Args:
        term (str): The term whose IDF is to be calculated.
        inverted_index (dict): A dictionary where keys are terms and values are lists of books containing those terms.
        total_relevant_documents (int): The number of relevant documents (titles containing any token from the query title).
        
        Returns:
        float: The IDF value of the term.
        """
        num_titles_with_term = len(inverted_index.get(term, [])) 
        if num_titles_with_term == 0:
            return 0
        return math.log(total_relevant_documents / num_titles_with_term)
    
    def compute_tfidf(self, tokens, inverted_index, total_relevant_documents, vocab=None):
        """
        Compute the TF-IDF values for a list of tokens using NumPy for efficiency.
        
        Args:
        tokens (list): A list of tokenized words from a title.
        inverted_index (dict): A dictionary where keys are terms and values are lists of books containing those terms.
        total_relevant_documents (int): The total number of relevant documents (books containing any token from the query title).
        vocab (list, optional): A list of all unique tokens (vocabulary) across the query and books for alignment. Defaults to None.
        
        Returns:
        np.array: A NumPy array of TF-IDF values for the tokens.
        """
        if vocab is None:
            vocab = list(set(tokens))

        term_count = Counter(tokens)
        total_terms = len(tokens)
        tf = np.zeros(len(vocab))
        
        token_to_index = {token: idx for idx, token in enumerate(vocab)}
        
        for term, count in term_count.items():
            if term in token_to_index:
                idx = token_to_index[term] 
                tf[idx] = count / total_terms 


        tfidf = np.zeros(len(vocab)) 
        for term in vocab:
            i = token_to_index[term] 
            idf = self.compute_idf(term, inverted_index, total_relevant_documents)
            tfidf[i] = tf[i] * idf
            
        return tfidf


    def cosine_similarity(self, query_tfidf, title_tfidf):
        """
        Compute the cosine similarity between two TF-IDF vectors using NumPy for vectorized operations.
        
        Args:
        query_tfidf (np.array): The TF-IDF vector for the query.
        title_tfidf (np.array): The TF-IDF vector for the title/document.
        
        Returns:
        float: The cosine similarity score between the query and the title.
        """
        dot_product = np.dot(query_tfidf, title_tfidf)  
        query_norm = np.linalg.norm(query_tfidf)  
        title_norm = np.linalg.norm(title_tfidf)  
        if query_norm == 0 or title_norm == 0:
            return 0
        return dot_product / (query_norm * title_norm) 

    def create_inverted_index_for_title(self, title):
        """
        Generate the inverted index for a given title using the `get_books_by_title_token` function.

        Args:
        title (str): The title for which to generate the inverted index.

        Returns:
        dict: The inverted index generated from the title, where each token maps to the list of relevant books containing that token.
        """
        tokenized_title = tokenize_text(title) 
        
        inverted_index = {}

        relevant_docs = set()

        for term in tokenized_title:
            books = self.dataset.get_books_by_title_token(term)
            inverted_index[term] = books
            relevant_docs.update(books)
        
        total_relevant_documents = len(relevant_docs)
        
        return inverted_index, relevant_docs, total_relevant_documents
    
    def remove_common_words_from_title(self, title: str, num_books,  frequency_threshold: float = 0.80,) -> str:
        """
        Removes common words from a book title based on their document frequency across the dataset.

        A word is considered "common" if it appears in more than `frequency_threshold` fraction of all titles 
        (e.g., 0.05 means remove words that appear in more than 5% of book titles).

        Args:
            title (str): The original book title.
            frequency_threshold (float): A value between 0 and 1 indicating the maximum allowed document frequency 
                                        for a word to be retained in the cleaned title.

        Returns:
            str: The input title with common words removed based on the specified frequency threshold.
        """
        def is_common(token):
            return self.title_vocab[token] / num_books >= frequency_threshold
        
        tokens = tokenize_text(title)
        new_tokens = []
        for token in tokens:
            if not is_common(token):
                new_tokens.append(token)
        return " ".join(new_tokens)
        
    def get_recs_from_title(self, title, top_n=20):
        """
        Get book recommendations based on the similarity to a given query title.
        
        Args:
        title (str): The book title provided by the user for similarity search.
        top_n (int, optional): The number of top similar titles to return. Defaults to 20.
        
        Returns:
        dict: A dictionary where the keys are book titles and the values are their similarity scores, sorted by similarity.
        """
        title = self.remove_common_words_from_title(title, self.dataset.num_books)

        inverted_index, relevant_docs, _ = self.create_inverted_index_for_title(title)
        tokenized_query = tokenize_text(title)
        
        vocab = list(set(tokenized_query))
        
        for book in relevant_docs:
            book_tokens = tokenize_text(book)
            vocab.extend(list(set(book_tokens))) 
        
        vocab = list(set(vocab)) 

        query_tfidf = self.compute_tfidf(tokenized_query, inverted_index, len(relevant_docs), vocab)
        
        similarity_scores = []
        for book in relevant_docs:
            book_tokens = tokenize_text(book)
            book_tfidf = self.compute_tfidf(book_tokens, inverted_index, len(relevant_docs), vocab)
            
            similarity_score = self.cosine_similarity(query_tfidf, book_tfidf)
            similarity_scores.append((book, similarity_score))
        
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_n_similar_titles = similarity_scores[:top_n]
        
        result_books = {title: score for title, score in top_n_similar_titles}
        
        return result_books
    
    def create_tfidf_matrix(self, books: dict) -> Tuple[List[str], csr_matrix, TfidfVectorizer]:
        """
        Generate a TF-IDF matrix from a dictionary of books
        
        Args:
            books (Dict[str, str]):
                A dictionary mapping each book title to its description text.

        Returns:
            Tuple[List[str], csr_matrix, TfidfVectorizer]:
                - A list of book titles, preserving the order corresponding to the TF-IDF matrix rows.
                - A sparse matrix of TF-IDF features (rows = books, columns = terms).
                - The trained TfidfVectorizer, which can be reused for transforming queries later.
        """
        titles: List[str] = []
        texts: List[str] = []

        for title, entries in books.items():
            if not entries:
                continue  

            book = entries[0]  
            desc = book.get("description", "")
            authors = " ".join(book.get("authors", []))
            categories = " ".join(book.get("categories", []))
            full_text = f"{title} {desc} {authors} {categories}".strip()

            titles.append(title)
            texts.append(full_text)

        vectorizer = TfidfVectorizer(stop_words="english", max_features=100_000)
        tfidf_matrix: csr_matrix = vectorizer.fit_transform(texts)

        return titles, tfidf_matrix, vectorizer

    def reduce_with_svd(self, tfidf_matrix: csr_matrix, n_components: int = NUM_LATENT_SEMANTIC_CONCEPTS) -> Tuple[np.ndarray, TruncatedSVD]:
        """
        Reduce the dimensionality of a TF-IDF matrix using Truncated Singular Value Decomposition (LSI).

        Args:
            tfidf_matrix (csr_matrix):
                The TF-IDF matrix of book descriptions (rows = books, columns = terms).
            n_components (int):
                The number of dimensions (latent semantic concepts) to retain.

        Returns:
            Tuple[np.ndarray, TruncatedSVD]:
                - A dense matrix of reduced-dimensional representations of books (shape: num_books × n_components).
                - The fitted TruncatedSVD object, for projecting future queries into the same space.
        """
        n_features = tfidf_matrix.shape[1]
        if n_components >= n_features:
            n_components = n_features - 1
            
        svd_model = TruncatedSVD(n_components)
        reduced_matrix = svd_model.fit_transform(tfidf_matrix)
        return reduced_matrix, svd_model

    def transform_query(self, query_text: str, vectorizer: TfidfVectorizer, svd: TruncatedSVD) -> np.ndarray:
        """
        Transforms a user query string into the same reduced-dimensional semantic space as the books.

        Args:
            query_text (str):
                The raw text of the user query (e.g., "scary and exciting").
            vectorizer (TfidfVectorizer):
                The trained vectorizer used on the book descriptions.
            svd (TruncatedSVD):
                The trained SVD transformer used to reduce the book vectors.

        Returns:
            np.ndarray:
                A 1D array representing the query in reduced semantic space (shape: 1 x n_components).
        """
        query_tfidf = vectorizer.transform([query_text])  # returns sparse vector
        reduced_query = svd.transform(query_tfidf)        # returns 1 x n_components dense vector

        return reduced_query

    def get_top_k_similar_books(self, query_vec: np.ndarray, book_vecs: np.ndarray, book_titles: List[str], k: int = 5) -> Dict[str, float]:
        """
        Compute cosine similarity between a query vector and all book vectors to retrieve the most similar titles.

        Args:
            query_vec (np.ndarray):
                The reduced-dimensional query vector (1 x n_components).
            book_vecs (np.ndarray):
                The reduced-dimensional matrix of book vectors (num_books x n_components).
            book_titles (List[str]):
                A list of book titles, ordered to match the rows of `book_vecs`.
            k (int):
                The number of top results to return.

        Returns:
            Dict[str, float]:
                A dictionary mapping top book titles to their cosine similarity scores.

        """
        cos_sims = cosine_similarity(query_vec, book_vecs)
        flattened_cos_sim_scores = cos_sims.flatten()
        sorted_indices = np.argsort(flattened_cos_sim_scores)[::-1]
        top_k_indices = sorted_indices[:k]
        top_matches = {book_titles[i]: flattened_cos_sim_scores[i] for i in top_k_indices}
        return top_matches


    def get_recs_by_description(self, description: str, title: str, authors: List[str], categories: List[str]) -> Dict[str, float]:
        """
        Generate book recommendations based on a combined semantic query from description, title, authors, and categories.

        Args:
            description (str):
                The main description or keywords of the book.
                This is the only required field and must not be empty.
            title (str):
                An optional title string to incorporate into the query.
            authors (List[str]):
                An optional list of author names.
            categories (List[str]):
                An optional list of category labels (e.g., genres or themes).

        Returns:
            Dict[str, float]:
                A dictionary mapping book titles to their similarity scores with the query,
                sorted in descending order of similarity.
        """
        description = tokenize_text(description)
        title = tokenize_text(title)
        authors = tokenize_name_list(authors)
        categories = tokenize_list(categories)

        composite_query = f"{description} {title} {' '.join(authors)} {' '.join(categories)}"
        composite_query = " ".join(composite_query.split())
        assert type(composite_query) == str

        query_vec = self.transform_query(composite_query, self.vectorizer, self.svd_model)
        if isinstance(self.book_vecs, TruncatedSVD):
            raise TypeError("book_vecs should be a matrix, not a model. Did you pass self.svd_model by mistake?")

        top_matches = self.get_top_k_similar_books(query_vec, self.book_vecs, self.book_titles)
        return top_matches

    
    def get_recommended_books(self, user_input, output_size=DEFAULT_RECS_SIZE,\
                               weights=DEFAULT_RECS_WEIGHTS) -> list[dict]:
        """    
            Returns a list of recommended books based on user input.
        """
        titles = user_input[INPUT_TITLES_KEY] 
        authors = user_input[INPUT_AUTHORS_KEY]
        categories = user_input[INPUT_CATEGORIES_KEY]

        title = ""
        if titles:
            title = titles[0] # TODO:Update to handle multiple titles

        title_recs = self.get_recs_from_title(title)
        author_recs = self.get_recs_from_author(authors)
        categ_recs = self.get_recs_from_categories(categories)

        aggregated_recs = self._aggregate_recs(title_recs, author_recs, categ_recs, weights)   
        top_sorted_recs = dict(sorted(aggregated_recs.items(), key=lambda x: x[1], reverse=True)[:output_size])
        final_recs = self._add_score_field(top_sorted_recs)

        return final_recs

    def _aggregate_recs(self, title_recs, author_recs, categs_recs, weights):

        title_recs = self._apply_weight(title_recs, weights.TITLES)
        author_recs = self._apply_weight(author_recs, weights.AUTHORS)
        categs_recs = self._apply_weight(categs_recs, weights.CATEGORIES)

        all_keys = set(title_recs.keys() | author_recs.keys() | categs_recs.keys())
        final_recs = {}
        for key in all_keys:
            t_score = title_recs.get(key, 0)
            a_score = author_recs.get(key, 0)
            c_score = categs_recs.get(key, 0)
            final_recs[key] = a_score + t_score + c_score 
        return final_recs
    
    def _apply_weight(self, recs, weight):
        for key, score in recs.items():
            updated_score = score * weight
            recs[key] = round(updated_score,2)
        return recs

    def _add_score_field(self, scores):
        books = []
        for title, score in scores.items():
            book = self.books[title][0] #TODO:Figure out how to handle books with same title in recommendation
            book[SCORE_KEY] = score
            books.append(book)
        return books
    