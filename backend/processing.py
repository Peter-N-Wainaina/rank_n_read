import math
import re
import numpy as np
from collections import Counter
from math import isnan

from .dataset import Dataset
from .utils import tokenize_text

class Processor(object):
    def __init__(self, json_file=None):
        dataset = Dataset()
        if json_file is not None:
            dataset = Dataset(json_file)
        self.dataset = dataset
        self.books = self.dataset.books
    
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
    
    def get_recs_from_categories(self, query: str):
        """
        Returns a list of book recommendations based on the query

        Parameters:
            query : list
                The input qeury list of categories
        Returns:
            dict:
                A dictionary of book titles and their scores based on categrories
        """
        # put the titles you get from Dataset.get_books_by_category in a set to take care of dups
        titles_by_category_set = set()
        for category in query:
            books_by_cat = self.dataset.get_books_by_category(category)
            if books_by_cat:
                titles_by_category_set.update(books_by_cat)
        
        # get the categories for each retrived subset and compute jaccard similarity
        # you should have a dict in the form {score: book infor dict...} after this block
        score_title_dict = {}
        query_categories_set = set(query)
        for title in titles_by_category_set:
            book_info = self.books[title][0]
            book_categories_set = set(book_info["categories"])
            sim_score = self.compute_jaccard_similarity(query_categories_set, book_categories_set)
            score_title_dict[title] = sim_score

        # sort the books in non-increasing order based on the similarity score

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

        # put the titles you get from Dataset.get_books_by_author in a set to take care of dups
        titles_by_authors_set = set()
        for author in authors:
            books_titles_by_author = self.dataset.get_books_by_author(author)
            if books_titles_by_author:
                titles_by_authors_set.update(books_titles_by_author)
        
        # make a set with the authors in it to pass into sim measure func
        authors_set = set(authors)

        # get the authors for each retrived subset and compute jaccard similarity btwn authors
        # you should have a dict in the form {score: book infor dict...} after this block
        score_books_dict = {}
        for title in titles_by_authors_set:
            book_info = self.books[title][0]
            book_authors_set = set(book_info["authors"])
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


    def get_recs_from_title(self, title, top_n=20):
        """
        Get book recommendations based on the similarity to a given query title.
        
        Args:
        title (str): The book title provided by the user for similarity search.
        top_n (int, optional): The number of top similar titles to return. Defaults to 20.
        
        Returns:
        dict: A dictionary where the keys are book titles and the values are their similarity scores, sorted by similarity.
        """
        inverted_index, relevant_docs,_   = self.create_inverted_index_for_title(title)
        
        tokenized_query = tokenize_text(title)  
        query_tfidf = self.compute_tfidf(tokenized_query, inverted_index)
        
        similarity_scores = []
        for book in relevant_docs:
            book_tokens = tokenize_text(book) 
            book_tfidf = self.compute_tfidf(book_tokens, inverted_index)
            
            similarity_score = self.cosine_similarity(query_tfidf, book_tfidf)
            similarity_scores.append((book, similarity_score))
        
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_n_similar_titles = similarity_scores[:top_n]
        
        result_books = {title: score for title, score in top_n_similar_titles}
        
        return result_books
    
    def get_recommended_books(self, user_input: dict[str: list[str]]) -> list[dict]:
        """    
            Returns a list of recommended books based on user input.
        """
        #TODO: Add recommendation functionality
        titles = user_input['titles']
        authors = user_input['authors']
        categories = user_input['categories']

        title = ""
        if titles:
            title = titles[0]

        fields = (title, authors, categories)
        funcs = (self.get_recs_from_title, self.get_recs_from_author, self.get_recs_from_categories)
        results = []

        for field, func in zip(fields, funcs):
            if field:
                results.append(func(field))
            else:
                results.append({})

        title_recs, author_recs, categ_recs = results

        merged = self._merge_recs(title_recs, author_recs, categ_recs, 0.6, 0.3, 0.1)
        sorted_recs = sorted(merged, key=lambda x: x[1])

        return_dict = []
        for rec, score in sorted_recs:
            rec_json = self.books[rec][0]
            rec_json["score"] = score

            new_json = {}
            for key, value in rec_json.items():
                if not (isinstance(value, str) or isinstance(value, list)):
                    new_json[key]  = ""
                else:
                    new_json[key] = value
            return_dict.append(new_json)

            

        return return_dict


    def _merge_recs(self, title_recs, author_recs, categ_recs, x, y, z):
        merged = []
        all_keys = set(title_recs) | set(author_recs) | set(categ_recs)
        
        for key in all_keys:
            t_score = title_recs.get(key, 0)
            a_score = author_recs.get(key, 0)
            c_score = categ_recs.get(key, 0)
            merged.append((key, x * t_score + y * a_score + z * c_score))
        return merged