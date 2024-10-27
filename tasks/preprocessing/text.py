import re
import unicodedata


class TextProcessing:
    def __init__(self,
                 lowercase=False,
                 strip_accents=False,
                 tokenizer=None,
                 stop_words=None,
                 keep_punctuation=True):
        """
        Initializes the TextProcessing class with customizable preprocessing options.

        Parameters:
            lowercase (bool): Whether to convert text to lowercase.
            strip_accents (bool): Whether to strip accents from text.
            tokenizer (callable): A function for tokenizing text (e.g., word_tokenize).
            stop_words (set): A set of stop words to remove from the tokens.
            keep_punctuation (bool): Whether to retain punctuation in the text.
        """
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.tokenizer = tokenizer if tokenizer else self.default_tokenizer
        self.stop_words = set(stop_words) if stop_words else set()
        self.keep_punctuation = keep_punctuation

    @staticmethod
    def default_tokenizer(text):
        """Splits text into words by whitespace."""
        return text.split()

    @staticmethod
    def remove_punctuation(text):
        """Removes punctuation from text."""
        return re.sub(r'[^\w\s]', '', text)

    @staticmethod
    def strip_accents_from_text(text):
        """Strips accents from characters in text."""
        return ''.join(
            char for char in unicodedata.normalize('NFD', text)
            if unicodedata.category(char) != 'Mn'
        )

    def remove_stopwords(self, tokens):
        """
        Removes stop words from a list of tokens.

        Parameters:
            tokens (list): List of tokens to process.

        Returns:
            list: Tokens with stop words removed.
        """
        return [word for word in tokens if word not in self.stop_words]

    def preprocess(self, text):
        """
        Applies all preprocessing steps to the text.

        Parameters:
            text (str): The input text to process.

        Returns:
            str: The fully preprocessed text.
        """
        if self.lowercase:
            text = text.lower()

        if self.strip_accents:
            text = self.strip_accents_from_text(text)

        if not self.keep_punctuation:
            text = self.remove_punctuation(text)

        tokens = self.tokenizer(text)
        tokens = self.remove_stopwords(tokens)

        return ' '.join(tokens)

    def __call__(self, text):
        return self.preprocess(text)


if __name__ == '__main__':
    txt = "How are    you."
    text_processor = TextProcessing(lowercase=True, keep_punctuation=False)
    text_processed = text_processor(txt)
    print("text_processed: ", text_processed)
