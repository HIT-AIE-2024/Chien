import nltk
from nltk.corpus import wordnet as wn
from typing import List, Optional

# Default tokenizer method
TOKENIZATION_METHOD = 'whitespace'

OptionalListStr = Optional[List[str]]
ListStr = List[str]

# Default 
TAG_DICT = {
    'a': wn.,
    's': ADJ_SAT,
    'r': ADV,
    'n': NOUN,
    'v': VERB
}

