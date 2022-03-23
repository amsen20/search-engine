import pandas as pd
import pickle
from tdata import *

df = pd.read_excel("IR1_7k_news.xlsx")

with open("positional-posting.pkl", "rb") as f:
    positional_posting = pickle.load(f)

while True:
    query = input("enter query: ")
    tokens = get_tokens(query)
    start_token = tokens[0]
    docs = {}
    docs_exact_occ = {}
    for token in set(tokens):
        for doc in positional_posting[token].positions:
            if doc not in docs:
                docs[doc] = 0
            docs[doc] += 1
    all_unq = len(set(tokens))
    for doc in docs:
        if docs[doc] < all_unq:
            docs_exact_occ[doc] = 0
            continue
        cnt = 0
        for pos in positional_posting[start_token].positions[doc]:
            f = True
            for ind, token in enumerate(tokens):
                if pos+ind not in positional_posting[token].positions[doc]:
                    f = False
                    break
            if f:
                cnt += 1

        docs_exact_occ[doc] = cnt

    def score(doc):
        return docs[doc] + docs_exact_occ[doc]

    results = list(docs)
    results = list(filter(lambda x: score(x) > 0, results))
    results.sort(key=score, reverse=True)
    results = results[:10]
    for ind, result in enumerate(results):
        print("rank: {}".format(ind))
        print(df["title"][result])
        print("score: {}".format(docs_exact_occ[result]))
        print("id: {}".format(result))
