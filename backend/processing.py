import re
import numpy as np
from collections import Counter

from .dataset import Dataset
from .utils import tokenize_text

class Processor(object):
    def __init__(self, json_file=None):
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
        Supports both exact matches (e.g., "J.K. Rowling") and partial matches (e.g., "Rowling").

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

    def process_title(self,title):
        """
        Tokenizes title into lowercase words and removes non-alphabet characters
        """
        title = title.lowe()
        title = re.sub(r'[^a-z\s]', '', title)
        return title.split()
    
    def title_to_vector(self, title, vocabulary):
        """
        convert a book's title into a vector based on the vocabulary
        """
        words = self.process_title(title)
        word_count = Counter(words)
        vector = [word_count.get(word, 0) for word in vocabulary]
        return vector
    
    def cosine_similarity(self, vec1, vec2):
        """
        compute cosine similarity between two vectors
        """
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)

        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product/ (magnitude1 * magnitude2)
    
    def find_similar_books(self, user_title, dataset):
        """
        Find the book with the highest similarity to the given user input using cosine similarity.
        
        Args:
            user_title (str): The title of the book input by the user.
            dataset (Dataset): The dataset object containing book data.

        Returns:
            str: The title of the book with the highest similarity to the user input.
        """
        vocabulary = set()
        for book in dataset.books.values():
            words = self.process_title(book['title'])
            vocabulary.update(words)

        vocabulary = sorted(vocabulary)

        #convert all book titles and user input to vectors
        user_vector = self.title_to_vector(user_title, vocabulary)
        books_vectors = {title: self.title_to_vector(title, vocabulary) for title in dataset.books.keys()}
        
        #compute cose similarity between user_title and all books
        similarities = {title: self.cosine_similarity(user_vector, book_vector) for title, book_vector in books_vectors.items()}

        #find the book with the highest similarity
        most_similarity_title = max(similarities, key = similarities.get)

        return most_similarity_title

    def get_recs_from_title(self, title, dataset, max_result = 50):
        """
        parameters
        ----------
        title: str
            The input title string. A string of a title of a book
        Returns
        -------
        list
            Alist of books recommendations

                Note: data -> {title: [{
                                "description": str,
                                "authors": ls,
                                "image": str,
                                "previewLink": str,
                                "publisher": str,
                                "publishedDate": str,
                                "categories": ls,
                                "title": str,
                                "similarity": int
                                }]}
        """
        most_similar_book_title = self.find_similar_books(title, dataset)
        most_similar_book = dataset.books[most_similar_book_title]

        #get the authors and the categories of the book
        authors = most_similar_book["authors"]
        categories = most_similar_book["categories"]

        #get books written by the same author
        books_by_author = []
        for author in authors:
            books_by_author.extend(self.get_recs_from_author(author))

        books_by_categories = self.get_recs_from_categories(dataset, ','.join(categories))

        #combine the results
        recomended_books = books_by_author + books_by_categories

        #rank the books by similarity score
        ranked_books = sorted(recomended_books, key = lambda x:x.get('similarity', 0), reverse=True)
        
        #return a max of max_results
        return ranked_books[:max_result]
    
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
