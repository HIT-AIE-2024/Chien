import nltk
from nltk.corpus import wordnet as wn
from typing import List, Optional


# Default tokenizer method
TOKENIZATION_METHOD = 'whitespace'

OptionalListStr = Optional[List[str]]
ListStr = List[str]

ADJ = wn.ADJ
ADJ_SAT = wn.ADJ_SAT
ADV = wn.ADV
NOUN = wn.NOUN
VERB = wn.VERB

TAG_DICT = {
    'a': ADJ,
    's': ADJ_SAT,
    'r': ADV,
    'n': wn.NOUN,
    'v': wn.VERB
}

