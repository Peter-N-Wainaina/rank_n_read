import json
from collections import defaultdict

from .config import DEFAULT_BOOKS_JSON_FILE
from .constants import CATEGORY_KEY, AUTHOR_KEY, DESCRIPTION_KEY
from .utils import tokenize_text, tokenize_name

class Dataset(object):
    def __init__(self, json_file=DEFAULT_BOOKS_JSON_FILE):
        self.books = self._load_books(json_file)
        self._book_title_to_id, self._book_id_to_title = self._build_book_index()
        self.authors_index = self._build_authors_index()
        self.categories_index = self._build_categories_index()
        self.titles_index = self._build_titles_index()
        self.title_vocab_frequency = self._build_title_vocab_frequency()
        self.book_data_dict = self._build_book_data_dict()
    
    @property
    def num_books(self) -> int:
        """
        Returns the number of books as a read-only property.
        """
        return len(self.books)

    def get_books_by_author(self, author_token: str) -> list[str]:
        """
        Retrieves a list of book titles written by an author. 
        Supports both exact matches (e.g., "J.K. Rowling") and partial matches (e.g., "Rowling").

        Args:
            author (str): The name of the author, which can be a full name or a partial match.

        Returns:
            list: A list of book titles associated with the given author.
                If no books are found, an empty list is returned.
        """
        if not author_token.strip():
            return []
        
        author_tokens = tokenize_name(author_token)
        book_ids = set()
        for author_token in author_tokens:
            author_book_ids = set(self.authors_index.get(author_token, []))
            book_ids = book_ids.union(author_book_ids)
        return self._get_book_titles_from_ids(list(book_ids))

    def get_books_by_category(self, category: str) -> list[str]:
        """
        Retrieves a list of book titles belonging to a specific category.
        Supports both exact matches (e.g., "Science Fiction") and partial matches (e.g., "Fiction").

        Args:
            category (str): The category name, which can be a full name or a partial match.

        Returns:
            list: A list of book titles associated with the given category.
                If no books are found, an empty list is returned.
        """
        if not category.strip():
            return []

        categ_tokens = tokenize_text(category)
        book_ids = set()
        for categ_token in categ_tokens:
            categ_book_ids = set(self.categories_index.get(categ_token, []))
            book_ids = book_ids.union(categ_book_ids)
        return self._get_book_titles_from_ids(list(book_ids))

    def get_books_by_title_token(self, token: str) -> list[str]:
        """
        Retrieves a list of book titles that contain the given token.

        The search is case-insensitive and matches any book title that includes the token
        when tokenized (e.g., "war" will match "The War of Art").

        Args:
            token (str): A lowercase or mixed-case token to search for within book titles.

        Returns:
            list[str]: A list of book titles that include the given token. 
                    Returns an empty list if no matches are found.
        """
        token = token.lower().strip()
        if not token or token not in self.titles_index:
            return []
        return [self._get_title_from_book_id(book_id) for book_id in self.titles_index[token]]
    
    def _get_book_titles_from_ids(self, books_ids: list[int]) -> list[str]:
        return [self._get_title_from_book_id(book_id) for book_id in books_ids]
    
    def _load_books(self, json_file: str) -> dict:
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
        return self._load_json_file(json_file)

    def _load_json_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def _build_authors_index(self):
        """
        Constructs an inverted index mapping authors to the books they have written
        and returns it.

        Returns:
            dict: An inverted index where:
                - Keys (str) represent author names.
                - Values (list) contain references to book entries associated with each author.
        """
        author_index = defaultdict(set)
        for title, entries in self.books.items():
            book_id = self._get_book_id_from_title(title)
            for entry in entries:
                raw_authors = entry.get(AUTHOR_KEY, [])

                tokenized_authors = []
                for author in raw_authors:
                    tokenized_authors.extend(tokenize_name(author))
                
                for author in tokenized_authors:
                    author_index[author].add(book_id)
        
        return {author: list(book_ids) for author, book_ids in author_index.items()}

    def _build_categories_index(self):
        """
        Constructs an inverted index mapping book categories (genres) to book IDs 
        of books that belong to them.

        Returns:
            dict: An inverted index where:
                - Keys (str) represent book categories.
                - Values (list) contain references to book entries within each category.
        """
        categ_index = defaultdict(set)
        for title, entries in self.books.items():
            book_id = self._get_book_id_from_title(title)
            for entry in entries:
                raw_categs = entry.get(CATEGORY_KEY, [])

                tokenized_categs= []
                for categ in raw_categs:
                    tokenized_categs.extend(tokenize_text(categ))
                
                for categ in tokenized_categs:
                    categ_index[categ].add(book_id)

        return {categ: list(book_ids) for categ, book_ids in categ_index.items()}
    
    def _build_book_index(self) -> tuple:
        """
        Builds mappings between book titles and unique indices.

        Returns:
            tuple: (dict, dict)
                - dict: A mapping of book titles to their unique index.
                - dict: A mapping of unique indices back to book titles.

        Raises:
            ValueError: If books are empty or None.
        """
        titles_list = sorted(list(self.books.keys()))
        book_title_to_id = {title:idx for idx, title in enumerate(titles_list)}
        book_id_to_title = {idx:title for idx, title in enumerate(titles_list)}

        return (book_title_to_id, book_id_to_title)

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
        if book_title not in self._book_title_to_id:
            raise KeyError(f"Book title '{book_title}' not in index")
        return self._book_title_to_id[book_title]

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
        if book_id not in self._book_id_to_title:
                raise KeyError(f"Book ID '{book_id}' not found in index.")

        return self._book_id_to_title[book_id]

    def _build_titles_index(self) -> dict[str, list[int]]:
        """
        Constructs and returns inverted index mapping tokens from book titles to book IDs.

        This method tokenizes each book title and creates a mapping from each lowercase
        token to the list of book IDs whose titles contain that token.

        Returns:
            dict[str, list[int]]: An inverted index where:
                - Keys are lowercase tokens from book titles.
                - Values are lists of book IDs (int) that contain the token in their title.
        """
        titles_index = defaultdict(set)
        for title in self.books.keys():
            book_id = self._get_book_id_from_title(title)
            tokens = tokenize_text(title)
            for token in tokens:
                titles_index[token].add(book_id)
        return {token: list(book_ids) for token, book_ids in titles_index.items()}
    
    def _build_title_vocab_frequency(self) -> dict[str, int]:
        """
        Constructs a vocabulary frequency dictionary from book titles.

        For each token (word) that appears in any book title, this function counts
        how many distinct book titles (i.e., documents) contain that token. The result 
        is a dictionary mapping each token to the number of titles it appears in.

        Returns:
            dict[str, int]: A dictionary where:
                - Keys are lowercase tokens extracted from book titles.
                - Values are integers representing how many unique titles contain each token.
        """
        vocab_freq = defaultdict(int)
        for title in self.books.keys():
            unique_tokens = set(tokenize_text(title))
            for token in unique_tokens:
                vocab_freq[token] += 1
        return vocab_freq

    def _build_book_data_dict(self) -> dict[str, str]:
        """
        Constructs a dictionary mapping book titles to their associated textual data.

        Args:
            raw_data (list of dict): A list of dictionaries where each dictionary contains 
                metadata for a single book (e.g., title, description, author, categories, etc).

        Returns:
            dict: A dictionary where the key is the book title (str) and the value is a 
                concatenated lowercase string of relevant book data (e.g. title + author + description + categories)

        """
        book_data_dict = defaultdict(str)
        for title,books in self.books.items():
            data_parts = []
            for book in books:
                description = book.get(DESCRIPTION_KEY, "")
                authors = " ".join(book.get(AUTHOR_KEY, []))
                categories = " ".join(book.get(CATEGORY_KEY, []))
                combined = f"{title} {description} {authors} {categories}"
                data_parts.append(combined.lower())

            book_data_dict[title] = " ".join(data_parts)  
          
        return book_data_dict
