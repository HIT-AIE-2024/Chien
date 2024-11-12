import nltk
from nltk.corpus import wordnet as wn
from typing import List, Optional

# Default tokenizer method
TOKENIZATION_METHOD = 'whitespace'

OptionalListStr = Optional[List[str]]
ListStr = List[str]

NOUN = w
TAG_DICT = {
    'a': wn.ADJ,
    's': wn.ADJ_SAT,
    'r': wn.ADV,
    'n': wn.NOUN,
    'v': wn.VERB
}

