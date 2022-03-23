import random

import tqdm
from math import log10, sqrt
from tdata import *
from math import *
from tf_idf import get_tf_idf_data
from gensim.models import Word2Vec
from numpy.linalg import norm
import pandas as pd
import numpy as np
import pickle

# path1: word2vec_model_hazm/w2v_150k_hazm_300_v2.model
# path2: custom_model/w2v_ir1.model
w2v_model = Word2Vec.load(input("Enter model path: "))
print("Model loaded.")
wh = int(input("Extract 7k or 50k? (7/50): "))
positional_path = "positional-posting.pkl" if wh == 50 else "positional-posting-7k.pkl"
with open(positional_path, "rb") as f:
    positional_posting = pickle.load(f)
print("Pickle loaded.")

# Extracting vectors:
idf, tf, _, _ = get_tf_idf_data(positional_posting)

doc_emb_vec = {}
doc_emb_ws = {}
for (token, data) in tqdm.tqdm(positional_posting.items()):
    for doc in data.positions:
        if doc not in doc_emb_vec:
            doc_emb_vec[doc] = np.zeros(300,)
            doc_emb_ws[doc] = 0
        weight = tf[token][doc] * idf[token]
        doc_emb_ws[doc] += weight
        try:
            doc_emb_vec[doc] += weight * w2v_model.wv[token]
        except KeyError:
            print(f"Ignoring: {token}")
            pass

for doc in doc_emb_vec:
    doc_emb_vec[doc] /= doc_emb_ws[doc]
print("Vectors calculated.")

out_path = "doc-emb-vec.pkl" if wh == 50 else "doc-emb-vec-7k.pkl"
with open(out_path, "wb") as f:
    pickle.dump(doc_emb_vec, f)
