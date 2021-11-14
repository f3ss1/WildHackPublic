import pickle

from natasha import (
    Doc,
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    MorphVocab
)
from pandas import read_csv
from navec import Navec
from tqdm import tqdm

PATH = 'navec_hudlit_v1_12B_500K_300d_100q.tar'  # Name of file for Navec

NAME = 'popularity'

# Natasha Setup.

segm = Segmenter()
_emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(_emb)
morph_vocab = MorphVocab()


def query_to_noun(query: str) -> list[str]:
    doc = Doc(query.lower())

    doc.segment(segmenter=segm)

    doc.tag_morph(morph_tagger)

    res_arr = []
    for token in doc.tokens:
        if token.pos == 'NOUN':
            token.lemmatize(morph_vocab)
            res_arr.append(token.lemma)

    return res_arr


# Navec setup.

navec = Navec.load(PATH)

# Data load.

data = read_csv('query_popularity.csv')
data.dropna(inplace=True)
data.reset_index(inplace=True)

pop_dict: dict[str, float] = {}
number_dict: dict[str, int] = {}

for i in tqdm(range(data.shape[0])):
    text = data.loc[i, 'query']
    text_popular = data.loc[i, 'query_popularity']
    noun_list = query_to_noun(text)
    for noun in noun_list:
        if noun in pop_dict:
            pop_dict[noun] += text_popular
            number_dict[noun] += 1
        else:
            pop_dict[noun] = text_popular
            number_dict[noun] = 1

for key in tqdm(pop_dict.keys()):
    pop_dict[key] /= number_dict[key]

# Dump.

with open(NAME + '.pkl', 'wb') as f:
    pickle.dump(pop_dict, f, pickle.HIGHEST_PROTOCOL)
