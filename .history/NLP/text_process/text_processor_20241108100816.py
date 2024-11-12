import nltk
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from config.configs import OptionalListStr, ListStr, TOKENIZATION_METHOD, TAG_DICT, NOUN


nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self) -> None:
        """
        Initializes the TextPreprocessor.
        """
        self.tag: str = None
        self.stemmed_tokens: OptionalListStr = None
        self.lemmatized_tokens: OptionalListStr = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def set_text(self, text: str) -> None:
        """
        Set the input text for preprocessing.

        Args:
            text (str): The input string that needs preprocessing.
        """
        self.text = text
    
    def tokenize(self, method: str = TOKENIZATION_METHOD) -> ListStr:
        """
        Tokenizes the input text using the specified method.

        Args:
            method (str, optional): The tokenization method to use. Options are 'whitespace', 'word_punct', or 'treebank'.
                                    Defaults to 'whitespace'.

        Raises:
            ValueError: If an invalid tokenization method is specified.

        Returns:
            ListStr: A list of tokens obtained after tokenization.
        """
        if self.text is None:
            raise ValueError("Input text not set. Please set the text using the set_text method.")
        
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
        self.tag = nltk.pos_tag([word])[0][1][0].lower() # Get the first character of the POS tag
        return TAG_DICT.get(self.tag, NOUN)
    
    def stemming(self, tokens) -> ListStr:
        """
        Performs stemming on the tokens.

        Raises:
            ValueError: If tokenization has not been performed.

        Returns:
            ListStr: A list of stemmed tokens.
        """
        if self.tokens is None:
            raise ValueError("Tokenization not performed. Please call the tokenize method first.")
        
        self.stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return self.stemmed_tokens
    
    def lemmatization(self, tokens) -> ListStr:
        """
        Performs lemmatization on the tokens.

        Raises:
            ValueError: If tokenization has not been performed.

        Returns:
            ListStr: A list of lemmatized tokens.
        """
        if self.tokens is None:
            raise ValueError("Tokenization not performed. Please call the tokenize method first.")
        
        self.lemmatized_tokens = [self.lemmatizer.lemmatize(token, pos=self.get_wordnet_pos(token)) for token in tokens]
        return self.lemmatized_tokens
