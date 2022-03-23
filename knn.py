import random
import tqdm
from math import *
from numpy.linalg import norm
import numpy as np
import pandas as pd
import pickle

MX_LEN = 100

dfs = [pd.read_excel("IR00_dataset_ph3/11k.xlsx"),
       pd.read_excel("IR00_dataset_ph3/17k.xlsx"),
       pd.read_excel("IR00_dataset_ph3/20k.xlsx")]
df = pd.concat(dfs, ignore_index=True)
print("Sheets loaded.")


with open("doc-emb-vec.pkl", "rb") as f:
    doc_emb_vec = pickle.load(f)
with open("doc-emb-vec-7k.pkl", "rb") as f:
    doc_emb_vec_7k = pickle.load(f)
with open("clusters.pkl", "rb") as f:
    clusters = pickle.load(f)
centers = clusters['centers']
idx = clusters['idx']
print("Pickles loaded.")


def get_sim(vec_a, vec_b):
    return np.dot(vec_a, vec_b)/(norm(vec_a) * norm(vec_b))


k = int(input("Enter k: "))


doc_to_cat = {}
for doc, vec in tqdm.tqdm(doc_emb_vec_7k.items()):
    cluster_ids = list(range(len(centers)))
    cluster_ids.sort(key=lambda x: get_sim(centers[x], vec), reverse=True)
    cluster_id = cluster_ids[0]
    classified_docs = list(idx[cluster_id])
    random.shuffle(classified_docs)
    classified_docs = classified_docs[:MX_LEN]
    classified_docs.sort(key=lambda c_d: get_sim(vec, doc_emb_vec[c_d]), reverse=True)
    cat_to_cnt = {}
    for c_d in classified_docs[:k]:
        cat = df["topic"][c_d]
        if cat not in cat_to_cnt:
            cat_to_cnt[cat] = 0
        cat_to_cnt[cat] += 1
    mx_cnt, max_cat = -1, None
    for cat, cnt in cat_to_cnt.items():
        if mx_cnt < cnt:
            mx_cnt = cnt
            max_cat = cat
    doc_to_cat[doc] = max_cat

with open("doc-to-cat.pkl", "wb") as f:
    pickle.dump(doc_to_cat, f)
