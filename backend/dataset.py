import os
from config import ROOT_PATH


DEFAULT_BOOKS_JSON_FILE = os.path.join(ROOT_PATH, 'data_exploration', 'books.json')

class Dataset(object):
    def __init__(self, json_file=DEFAULT_BOOKS_JSON_FILE):
        self.books = self._load_books(json_file)
        self.authors_index = self._build_authors_index()
        self.categories_index = self._build_categories_index()
        self._book_title_to_id, self._book_id_to_title = self._build_book_index(self.books)

    def get_books_by_author(self, author: str):
        """
        Retrieves a list of book titles written by an author. 
        Supports both exact matches (e.g., "J.K. Rowling") and partial matches (e.g., "Rowling").

        Args:
            author (str): The name of the author, which can be a full name or a partial match.

        Returns:
            list: A list of book titles associated with the given author.
                If no books are found, an empty list is returned.
        """
        raise NotImplementedError

    def get_books_by_category(self, category: str):
        """
        Retrieves a list of book titles belonging to a specific category.
        Supports both exact matches (e.g., "Science Fiction") and partial matches (e.g., "Fiction").

        Args:
            category (str): The category name, which can be a full name or a partial match.

        Returns:
            list: A list of book titles associated with the given category.
                If no books are found, an empty list is returned.
        """
        raise NotImplementedError
    
    def _load_books(self, json_file: str):
        """
        Reads a books from a JSON file and returns them as a dictionary.

        Args:
            json_file (str): The path to the JSON file containing book data.

        Returns:
            dict: A dictionary where keys are book titles (str) and values are lists 
                of dictionaries containing book details:
                - `description` (str): Summary of the book.
                - `authors` (list of str): List of authors.
                - `image` (str): URL of the book cover image.
                - `previewLink` (str): URL to preview the book.
                - `publisher` (str): Name of the publisher.
                - `publishedDate` (str): Date of publication.
                - `categories` (list of str): Categories or genres.
                - `title` (str): Full title of the book.
        """
        raise NotImplementedError

    def _build_authors_index(self):
        """
        Constructs an inverted index mapping authors to the books they have written
        and returns it.

        Returns:
            dict: An inverted index where:
                - Keys (str) represent author names.
                - Values (list) contain references to book entries associated with each author.
        """
        raise NotImplementedError

    def _build_categories_index(self):
        """
        Constructs an inverted index mapping book categories (genres) to books that belong to them
        and returns it.

        Returns:
            dict: An inverted index where:
                - Keys (str) represent book categories.
                - Values (list) contain references to book entries within each category.
        """
        raise NotImplementedError
    
    def _build_book_index(self, books: dict) -> tuple:
        """
        Builds mappings between book titles and unique indices.

        Args:
            books (dict): A dictionary containing book titles as keys.

        Returns:
            tuple: (dict, dict)
                - dict: A mapping of book titles to their unique index.
                - dict: A mapping of unique indices back to book titles.

        Raises:
            ValueError: If books are empty or None.
        """
        raise NotImplementedError

    def _get_book_id_from_title(self, book_title: str) -> int:
        """
        Retrieves the unique index assigned to a given book title.

        Args:
            book_title (str): The title of the book.

        Returns:
            int: The unique index of the book.

        Raises:
            KeyError: If the book title is not found.
        """
        raise NotImplementedError

    def _get_title_from_book_id(self, book_id: int) -> str:
        """
        Retrieves the book title corresponding to the given unique index.

        Args:
            book_id (int): The unique index assigned to the book.

        Returns:
            str: The title of the book.

        Raises:
            KeyError: If the book ID is not found.
        """
        raise NotImplementedError
