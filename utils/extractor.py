import string

from nltk import ngrams, word_tokenize

translator = str.maketrans('', '', string.punctuation)


def tokenize(text: str):
    return word_tokenize(text.lower().translate(translator).strip())


def is_substring(text: str, elements: set) -> bool:
    """
    Check if a string is a substring of any string in a set

    Args:
        text (str): text to be tested
        elements (set(str)): set of string to be tested against for substring condition

    Return:
        (bool): whether or not if text is a substring of strings in elements
    """
    for element in elements:
        if text in element:
            return True

    return False


def get_extractor(token_extractor_type: str):
    if token_extractor_type == 'any':
        return any_substring_extraction
    return max_substring_extraction


def max_substring_extraction(text: str, entities: set, ngram: int = 3, stopwords: set = None) -> set:
        """
        The function extract all valid entities based on maximum substring policy

        Args:
            text (str): string from which entities to be extracted
            entities (set): set of all valid entities
            n (int): size of n-gram
            stopwords (set): set of words to be ignored
        Returns:
            set(str): set containing extracted entitites
        """
        candidates = set()

        # Iterative build N-grams with reducing N and preserve biggest ngram entities
        for N in range(ngram, 0, -1):
            for tokens in ngrams(tokenize(text), N):
                entity = '_'.join(tokens)
                if entity in entities:
                    if (stopwords and entity in stopwords.words()) or is_substring(entity, candidates) :
                        continue
                    candidates.add(entity)

        return candidates


def any_substring_extraction(text: str, entities: set, ngram: int = 3, stopwords: set = None) -> set:
    candidates = set()
    for N in range(ngram, 0, -1):
        for tokens in ngrams(tokenize(text), N):
            entity = '_'.join(tokens)
            if entity in entities:
                if stopwords and entity in stopwords:
                    continue
                candidates.add(entity)

    return candidates
