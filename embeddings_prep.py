# Imports & Constants.
import pickle

import numpy as np
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

CHUNK_SIZE = 500_000  # Chunk size
N_CHUNKS = 4  # Number of chunks.

PATH = 'navec_hudlit_v1_12B_500K_300d_100q.tar'  # Name of file for Navec

NAME = 'embeddings'

# Uncomment below if Navec's not set up.
# !python -m wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar

# Dataset is large so we go chunks.

data_marked = read_csv('search_history.csv', chunksize=CHUNK_SIZE)

# Natasha setup.

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

# Creating possible tags with their embeddings.

embed_dict: dict[str, np.ndarray] = {}

for i, chunk in enumerate(data_marked):
    if i >= N_CHUNKS:
        break
    print(f'Chunk {i + 1}/{N_CHUNKS}')
    chunk.dropna(inplace=True)
    chunk.reset_index(inplace=True)
    for j in tqdm(range(chunk.shape[0])):
        text = chunk.loc[j, 'UQ']
        noun_list = query_to_noun(text)
        for noun in noun_list:
            if noun not in embed_dict and noun in navec:
                embed_dict[noun] = navec[noun]

# Dump.

with open(NAME + '.pkl', 'wb') as f:
    pickle.dump(embed_dict, f, pickle.HIGHEST_PROTOCOL)
