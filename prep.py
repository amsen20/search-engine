import pandas as pd
import pickle
from tdata import TokenData, get_tokens
import tqdm


dfs = [pd.read_excel("IR00_dataset_ph3/11k.xlsx"),
       pd.read_excel("IR00_dataset_ph3/17k.xlsx"),
       pd.read_excel("IR00_dataset_ph3/20k.xlsx")]
df = pd.concat(dfs, ignore_index=True)

sz = df[df.columns[0]].count()

positional_posting = {}

for i in tqdm.tqdm(range(sz)):
    tokens = get_tokens(df['content'][i])

    for pos, token in enumerate(tokens):
        if token not in positional_posting:
            positional_posting[token] = TokenData()

        positional_posting[token].add(i, pos)

with open("positional-posting.pkl", "wb") as f:
    pickle.dump(positional_posting, f)


