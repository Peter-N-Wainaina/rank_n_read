import math
import numpy as np
from collections import Counter

from .dataset import Dataset
from .utils import tokenize_text, tokenize_name
from .constants import DEFAULT_RECS_WEIGHTS, SCORE_KEY, INPUT_AUTHORS_KEY,\
    INPUT_CATEGORIES_KEY, INPUT_TITLES_KEY, DEFAULT_RECS_SIZE

class Processor(object):
    def __init__(self, json_file=None):
        dataset = Dataset()
        if json_file is not None:
            dataset = Dataset(json_file)
        self.dataset = dataset
        self.books = self.dataset.books
        self.title_vocab = self.dataset.title_vocab_frequency

    
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
        modified_query = []
        for compound_query in query:
            ls = tokenize_text(compound_query)
            modified_query.extend(ls)
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
            book_categories_ls = []
            for comp_category in book_info["categories"]:
                ls = tokenize_text(comp_category)
                book_categories_ls.extend(ls)
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
        modified_authors = []
        for author in authors:
            ls = tokenize_name(author)
            modified_authors.extend(ls)
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
            modified_authors_from_book = []
            for author in book_info["authors"]:
                ls = tokenize_name(author)
                modified_authors_from_book.extend(ls)
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
    