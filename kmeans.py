import random
import tqdm
from math import *
from numpy.linalg import norm
import numpy as np
import pickle


with open("doc-emb-vec.pkl", "rb") as f:
    doc_emb_vec = pickle.load(f)


def get_sim(vec_a, vec_b):
    return np.dot(vec_a, vec_b)/(norm(vec_a) * norm(vec_b))


K = int(input("Enter k: "))
docs = list(doc_emb_vec)


def initialize_centers():
    vecs = [doc_emb_vec[doc] for doc in docs]
    return random.sample(vecs, K)


def find_closest_centers(centers):
    idx = [set() for _ in range(len(centers))]

    for doc in docs:
        mx_sim, mx_cen = -inf, -1
        for j, cen in enumerate(centers):
            sim = get_sim(doc_emb_vec[doc], cen)
            if sim > mx_sim:
                mx_sim = sim
                mx_cen = j
        idx[mx_cen].add(doc)
    return idx


def compute_means(idx):
    centers = []
    for i in range(len(idx)):
        pt = np.zeros(shape=doc_emb_vec[docs[0]].shape)
        for doc in idx[i]:
            pt += doc_emb_vec[doc]
        pt /= len(idx[i])
        centers.append(pt)
    return centers


cs = initialize_centers()
for _ in tqdm.tqdm(range(15)):
    idx = find_closest_centers(cs)
    cs = compute_means(idx)

idx = find_closest_centers(cs)

RSS = 0
for i in range(len(idx)):
    for doc in idx[i]:
        RSS += get_sim(cs[i], doc_emb_vec[doc])
print(f"RSS: {RSS}")

clusters = {
    "idx": idx,
    "centers": cs
}
with open("clusters.pkl", "wb") as f:
    pickle.dump(clusters, f)
