import nltk
from nltk.corpus import wordnet as wn
from typing import List, Optional

# Download NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Default tokenizer method
TOKENIZATION_METHOD = 'whitespace'

OptionalListStr = Optional[List[str]]
ListStr = List[str]
