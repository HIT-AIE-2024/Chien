import nltk
from nltk.corpus import wordnet as wn
from typing import List, Optional

# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

WORDNET = 'wordnet'


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
    'n': NOUN,
    'v': VERB
}

