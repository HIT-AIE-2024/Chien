import nltk
from nltk.corpus import wordnet as wn
from typing import List, Optional

# Default tokenizer method
TOKENIZATION_METHOD = 'whitespace'

OptionalListStr = Optional[List[str]]
ListStr = List[str]

# Default 
ADJ = wn.ADJ
ADJ_SAT = wn.ADJ_SAT
ADV = wn.ADV
NOUN = wn.NOUN
VERB = wn.VERB

