import re
import numpy as np
from collections import Counter

from .dataset import Dataset
from .utils import tokenize_text

class Processing(object):
    def __init__(self):
        self.dataset = Dataset()
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
        return intersection_set / union if union != 0 else 0
    
    def get_recs_from_categories(self, query: str):
        """
        Returns a list of book recommendations based on the query

        Parameters:
            query : str
                The input qeury string. This is expected to be comma separated categories
        Returns:
            list:
                A list of book recommendations.
        """
        # tokenize the query
        # for each category 
            # get a list of books associated with that category
            # add these book titles to a set
        # now you have a set of all the books for the categories you got
        # for the subset of books that you have, order them by similarity measure ie jaccard 
        # return the ordered list...not too sure if I should return only a 
        # specific number of books?

        # get a list of categories from the query
        categories_ls = tokenize_text(query)

        # put the titles you get from Dataset.get_books_by_category in a set to take care of dups
        titles_by_category_set = set()
        for category in categories_ls:
            books_by_cat = self.dataset.get_books_by_category(category)
            if books_by_cat:
                titles_by_category_set.update(books_by_cat)
        
        # get the categories for each retrived subset and compute jaccard similarity
        # you should have a dict in the form {score: book infor dict...} after this block
        score_books_dict = {}
        query_categories_set = set(categories_ls)
        for title in titles_by_category_set:
            book_info = self.books[title]
            book_categories_set = set(book_info["categories"])
            sim_score = self.compute_jaccard_similarity(query_categories_set, book_categories_set)
            book_info["score"] = sim_score
            score_books_dict[sim_score] = book_info

        # sort the books in non-increasing order based on the similarity score
        sorted_result = [book_info for _, book_info in sorted(score_books_dict.items(), key=lambda item: item[0], reverse=True)]

        return sorted_result

    def get_recs_from_author(self, author: str):
        """
        Returns a list of recommended books based on the author

        Parameters:
            author: str
                This is the input query. We are doing one author for now. We might
                add more authors later

        Returns:
            list:
                A list of book recommendations
        """

        # get the books associated with this author
        titles_assoc_with_author = self.dataset.get_books_by_author(author)
        
        # make a set with just the author in it to pass into sim measure func
        author_set = {author}

        # get the authors for each retrived subset and compute jaccard similarity btwn authors
        # you should have a dict in the form {score: book infor dict...} after this block
        score_books_dict = {}
        for title in titles_assoc_with_author:
            book_info = self.books[title]
            book_authors_set = set()
            book_authors_set.update(book_info["authors"])
            sim_score = self.compute_jaccard_similarity(author_set, book_authors_set)
            book_info["score"] = sim_score
            score_books_dict[sim_score] = book_info
        
        # sort the books in non-increasing order based on the similarity score
        sorted_result = [book_info for _, book_info in sorted(score_books_dict.items(), key=lambda item: item[0], reverse=True)]

        return sorted_result

    def get_recs_from_title(title: str, top_n=20) -> dict:
        """
        Computes the similarity score for books based on a given title.

        This function tokenizes the user input title, retrieves the books that 
        contain the tokens, counts how many times each book appears, and then 
        calculates and normalizes the similarity score. It then returns the top 
        N books with the highest similarity scores.

        Args:
            title (str): The title of the book entered by the user. This is the 
                        input title that will be tokenized and compared against 
                        other books in the dataset.
            top_n (int, optional): The number of top books to return based on their 
                                similarity scores. Default is 20.

        Returns:
            dict: A dictionary containing the book titles as keys and their 
                normalized similarity scores as values. The dictionary is 
                limited to the top N books with the highest similarity scores.
        """

        tokens = tokenize_text(title)
        
        combined_list = []
        for token in tokens:
            books = Dataset.get_books_by_title_token(token)
            combined_list.extend(books) 
        
        book_counts = Counter(combined_list) 
        
        total_books = len(combined_list)
        normalized_scores = {book: count / total_books for book, count in book_counts.items()}
        
        sorted_books = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_books = dict(sorted_books[:top_n])
        
        return top_books
    
    def get_recommended_books(self, user_input: dict[str: list[str]]) -> list[dict]:
        """    
            Returns a list of recommended books based on user input.
        """
        #TODO: Add recommendation functionality
        place_holder_books = []
        for _, entries in self.books.items():
            place_holder_books.extend(entries)
            if len(place_holder_books) >= 5:
                break
        return place_holder_books
