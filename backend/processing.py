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
