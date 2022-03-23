from math import log10, sqrt
from tdata import *
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
df = pd.read_excel("IR1_7k_news.xlsx")


with open("positional-posting.pkl", "rb") as f:
    positional_posting = pickle.load(f)

K = 10  # number of returned docs

idf, tf, _, _ = get_tf_idf_data(positional_posting)

doc_emb_vec = {}
doc_emb_ws = {}
for (token, data) in positional_posting.items():
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

while True:
    query = input("enter query: ")
    tokens = get_tokens(query)
    q_emb = np.zeros(300,)
    q_ws = 0
    for token in set(tokens):
        c_qtf = 1 + log10(tokens.count(token))
        weight = c_qtf * idf[token]
        q_ws += weight
        try:
            q_emb += w2v_model.wv[token] * weight
        except KeyError:
            print(f"Ignoring: {token}")
            pass

    q_emb /= q_ws

    results = list(doc_emb_vec)

    def cos_emb(doc):
        d_emb = doc_emb_vec[doc]
        return np.dot(q_emb, d_emb)/(norm(q_emb) * norm(d_emb))

    results.sort(key=cos_emb, reverse=True)
    results = results[:K]
    for ind, result in enumerate(results):
        print("rank: {}".format(ind + 1))
        print("id: {}".format(result))
        print(df["title"][result])
