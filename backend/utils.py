import re

def tokenize_text(text:str) -> list[str]:
    """
    Tokenizes a text string into lowercase word tokens.

    This function is intended for general-purpose text like categories, tags,
    and other non-name-based fields. It strips punctuation, lowercases the text,
    and extracts only alphabetic word tokens.

    Args:
        text (str): The input string to tokenize.

    Returns:
        list[str]: A list of lowercase tokens extracted from the input.
    """
    return re.findall(r"[a-z]+", text.lower())

def tokenize_name(name: str) -> list[str]:
    """
    Tokenizes a name, preserving dot-separated initials (e.g., 'J.K.') as a single token.

    Args:
        text (str): Author name as a string.

    Returns:
        list[str]: List of lowercase name tokens.
    """
    cleaned = re.sub(r"\s+", " ", name.strip().lower())     # Collapse multiple spaces and normalize
    return re.findall(r"[a-z]+'[a-z]+|[a-z]+(?:\.[a-z]+)*", cleaned)     # Match dot-separated initials and words (e.g. 'j.k.', 'rowling', "o'connor")

def tokenize_name_list(authors_ls: list[str]) -> list[str]:
    """
    Tokenizes a list of names, preserving dot-separated initials (e.g., 'J.K.') as a single token.

    Args:
        authors_ls (list[str]): List of author names

    Returns:
        list[str]: List of tokenized author names
    """
    tokenized_ls = []
    for name in authors_ls:
        tokenized_ls.extend(tokenize_name(name))
    return tokenized_ls

def tokenize_list(ls: list[str]) -> list[str]:
    """
    Tokenizes a list

    Args:
        authors_ls (list[str]): List strings

    Returns:
        list[str]: List of tokenized strings
    """
    tokenized_ls = []
    for item in ls:
        tokenized_ls.extend(tokenize_text(item))
    return tokenized_ls
