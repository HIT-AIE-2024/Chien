import nltk
from nltk.corpus import wordnet as wn
from typing import List, Optional

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

TOKENIZATION_METHOD = 'whitespace'
