from math import log10, sqrt

KCH = 20  # number of champions for each term
N = 100000  # constant for tf-idf


def get_tf_idf_data(positional_posting):
    idf = {}
    tf = {}
    length = {}
    champions = {}

    for (token, data) in positional_posting.items():
        idf[token] = log10(N / len(data.positions))
        if token not in tf:
            tf[token] = {}

        ls = list(data.positions)
        ls.sort(key=lambda doc: len(data.positions[doc]), reverse=True)
        ls = ls[:KCH]
        champions[token] = ls

        for doc in data.positions:
            f = len(data.positions[doc])
            val = 1 + log10(len(data.positions[doc]))
            tf[token][doc] = val
            if doc not in length:
                length[doc] = 0
            length[doc] += val ** 2

    return idf, tf, length, champions
