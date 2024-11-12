import nltk
from nltk.corpus import wordnet as wn
from typing import List, Optional

# Download NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Default tokenizer method
TOKENIZATION_METHOD = 'whitespace'

# Define a type alias for Optional[List[str]]
OptionalListStr = Optional[List[str]]
