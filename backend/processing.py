import re
import numpy as np
from collections import Counter
from math import isnan

from .dataset import Dataset
from .utils import tokenize_text, tokenize_name

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

    def get_recs_from_title(self, title: str, top_n=20) -> dict:
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
            books = self.dataset.get_books_by_title_token(token)
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