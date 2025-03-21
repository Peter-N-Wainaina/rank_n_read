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
