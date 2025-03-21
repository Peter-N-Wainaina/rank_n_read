import re
import json
import pandas as pd
from collections import defaultdict

class Processing:
    def __init__(self, json_file_path):
        self.data = self.load_data(json_file_path)
        self.books_df = self.create_dataframe()

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
        
    def create_dataframe(self):
        books_data = []
        for book_title in self.data:
            book_details = self.data[book_title]
            for book in book_details:
                books_data.append(book)
        return pd.DataFrame(books_data)

    def get_books(self, num_books=10):
        sample_books = self.books_df.head(num_books)
        return sample_books.to_json(orient='records')

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
    
    def get_recs_from_categories(self, query):
        """
        Parameters
        ----------
        query : str
            The input qeury string. This is expected to be comma separated categories
        Returns
        -------
        list
            A list of book recommendations.

        Note: data -> {title: [{
                                "description": str,
                                "authors": ls,
                                "image": str,
                                "previewLink": str,
                                "publisher": str,
                                "publishedDate": str,
                                "categories": ls,
                                "title": str
                                }]}
        """
        pass
