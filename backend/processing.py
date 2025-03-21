import re
import json
import pandas as pd
from collections import defaultdict
from dataset import Dataset

class Processing(object):
    def __init__(self):
        self.dataset = Dataset()
        self.books = self.dataset.books

    def tokenize(self, text):
        """Returns a list of words that make up the text.    

        Parameters
        ----------
        text : str
            The input text string

        Returns
        -------
        list
            A list of tokens corresponding to the input string.
        """
        return [x for x in re.findall(r"[a-z]+", text.lower())]
    
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
        categories_ls = self.tokenize(query)

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
            score_books_dict[sim_score] = book_info
        
        # sort the books in non-increasing order based on the similarity score
        sorted_result = [book_info for _, book_info in sorted(score_books_dict.items(), key=lambda item: item[0], reverse=True)]

        return sorted_result