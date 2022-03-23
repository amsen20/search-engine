import time

from math import log10, sqrt
from tdata import *
from tf_idf import get_tf_idf_data
import pandas as pd
import pickle

df = pd.read_excel("IR1_7k_news.xlsx")


with open("positional-posting-7k.pkl", "rb") as f:
    positional_posting = pickle.load(f)
with open("doc-to-cat.pkl", "rb") as f:
    doc_to_cat = pickle.load(f)

K = 5  # number of returned docs

idf, tf, length, champions = get_tf_idf_data(positional_posting)

while True:
    query = input("enter query: ")
    cat = input("enter cat: ")
    tokens = get_tokens(query)
    qtf = {}
    docs = set()
    for token in set(tokens):
        qtf[token] = 1 + log10(tokens.count(token))
        for doc in positional_posting[token].positions:
            if cat == "any" or cat == doc_to_cat[doc]:
                docs.add(doc)

    score = {}
    for token in set(tokens):
        if token not in positional_posting:
            continue

        for doc in docs:
            if doc not in score:
                score[doc] = 0
            if doc in tf[token]:
                score[doc] += qtf[token] * tf[token][doc] * idf[token]

    for doc in score:
        if doc in length and length[doc] > 0:
            score[doc] /= sqrt(length[doc])

    results = list(score)
    results = list(filter(lambda x: score[x] > 0, results))
    results.sort(key=lambda x: score[x], reverse=True)
    results = results[:K]
    for ind, result in enumerate(results):
        print("rank: {}".format(ind+1))
        print("score: {}".format(score[result]))
        print("id: {}".format(result))
        print(df["title"][result])

