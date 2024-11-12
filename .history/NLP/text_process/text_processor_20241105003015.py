import nltk
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from config.configs import ListStr, TOKENIZATION_METHOD

class TextPreprocessor:
    def __init__(self, text: str) -> None:
        """
        Initializes the TextPreprocessor with the input text.

        Args:
            text (str): The input string that needs preprocessing.
        """
        self.text: str = text
        self.tokens: ListStr = None
        self.stemmed_tokens: ListStr = None
        self.lemmatized_tokens: ListStr = None
    
    def tokenize(self, method: str = 'whitespace') -> List[str]:
        """
        Tokenizes the input text using the specified method.

        Args:
            method (str, optional): The tokenization method to use. Options are 'whitespace', 'word_punct', or 'treebank'.
                                    Defaults to 'whitespace'.

        Raises:
            ValueError: If an invalid tokenization method is specified.

        Returns:
            List[str]: A list of tokens obtained after tokenization.
        """
        if method == 'whitespace':
            tokenizer = WhitespaceTokenizer()
        elif method == 'word_punct':
            tokenizer = WordPunctTokenizer()
        elif method == 'treebank':
            tokenizer = TreebankWordTokenizer()
        else:
            raise ValueError("Invalid tokenization method. Choose 'whitespace', 'word_punct', or 'treebank'.")

        self.tokens = tokenizer.tokenize(self.text)
        return self.tokens
    
    def get_wordnet_pos(self, word: str) -> str:
        """
        Converts POS tags to WordNet format.

        Args:
            word (str): The word for which to determine the part of speech.

        Returns:
            str: The corresponding WordNet POS tag.
                 Returns 'n' for noun, 'v' for verb, 'a' for adjective,
                 'r' for adverb, or defaults to 'n' (noun) if the tag is unknown.
        """
        tag = nltk.pos_tag([word])[0][1][0].lower()  # Get the first character of the POS tag
        tag_dict = {
            'a': wn.ADJ,       # adjective
            's': wn.ADJ_SAT,   # satellite adjective
            'r': wn.ADV,       # adverb
            'n': wn.NOUN,      # noun
            'v': wn.VERB       # verb
        }
        return tag_dict.get(tag, wn.NOUN)
    
    def stemming(self, tokens: List[str]) -> List[str]:
        """
        Performs stemming on the given tokens.

        Args:
            tokens (List[str]): A list of tokens to perform stemming on.

        Raises:
            ValueError: If tokenization has not been performed before stemming.

        Returns:
            List[str]: A list of stemmed tokens.
        """
        if self.tokens is None:
            raise ValueError("Tokenization not performed. Please call the tokenize method first.")
        
        stemmer = PorterStemmer()
        self.stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return self.stemmed_tokens
    
    def lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Performs lemmatization on the given tokens.

        Args:
            tokens (List[str]): A list of tokens to perform lemmatization on.

        Raises:
            ValueError: If tokenization has not been performed before lemmatization.

        Returns:
            List[str]: A list of lemmatized tokens.
        """
        if self.tokens is None:
            raise ValueError("Tokenization not performed. Please call the tokenize method first.")
        
        lemmatizer = WordNetLemmatizer()
        self.lemmatized_tokens = [lemmatizer.lemmatize(token, pos=self.get_wordnet_pos(token)) for token in tokens]
        return self.lemmatized_tokens
