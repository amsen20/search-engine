from math import log10, sqrt
from tdata import *
from tf_idf import get_tf_idf_data
from gensim.models import Word2Vec
from numpy.linalg import norm
import pandas as pd
import numpy as np
import pickle
import time

# path1: word2vec_model_hazm/w2v_150k_hazm_300_v2.model
# path2: custom_model/w2v_ir1.model
w2v_model = Word2Vec.load(input("Enter model path: "))
print("Model loaded.")
dfs = [pd.read_excel("IR00_dataset_ph3/11k.xlsx"),
       pd.read_excel("IR00_dataset_ph3/17k.xlsx"),
       pd.read_excel("IR00_dataset_ph3/20k.xlsx")]
df = pd.concat(dfs, ignore_index=True)
print("Sheets loaded.")


with open("positional-posting.pkl", "rb") as f:
    positional_posting = pickle.load(f)
with open("clusters.pkl", "rb") as f:
    clusters = pickle.load(f)
with open("doc-emb-vec.pkl", "rb") as f:
    doc_emb_vec = pickle.load(f)
centers = clusters['centers']
idx = clusters['idx']
print("Got objects.")

K = 10  # number of returned docs
b = int(input("Enter b: "))
idf, tf, _, _ = get_tf_idf_data(positional_posting)


def get_sim(vec_a, vec_b):
    return np.dot(vec_a, vec_b)/(norm(vec_a) * norm(vec_b))


while True:
    query = input("enter query: ")
    start = time.time()
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

    cluster_ids = list(range(len(centers)))
    cluster_ids.sort(key=lambda x: get_sim(centers[x], q_emb), reverse=True)
    cluster_ids = cluster_ids[:b]

    results = []
    for cluster_id in cluster_ids:
        results += idx[cluster_id]

    results.sort(key=lambda doc: get_sim(doc_emb_vec[doc], q_emb), reverse=True)
    results = results[:K]
    print(f"time: {time.time() - start}")
    for ind, result in enumerate(results):
        print("rank: {}".format(ind + 1))
        print("id: {}".format(result))
        print(df["url"][result])
