import multiprocessing
import time
import tqdm
import pandas as pd
from tdata import get_tokens
from gensim.models import Word2Vec

dfs = [pd.read_excel("IR00_dataset_ph3/11k.xlsx"),
       pd.read_excel("IR00_dataset_ph3/17k.xlsx"),
       pd.read_excel("IR00_dataset_ph3/20k.xlsx")]
df = pd.concat(dfs, ignore_index=True)
sz = df[df.columns[0]].count()

training_data = []
for i in tqdm.tqdm(range(sz)):
    tokens = get_tokens(df['content'][i])
    training_data.append(tokens)

w2v_model = Word2Vec(min_count=1,
                     window=5,
                     vector_size=300,
                     alpha=0.03,
                     workers=multiprocessing.cpu_count()-1)

w2v_model.build_vocab(training_data)

start = time.time()
w2v_model.train(training_data, total_examples=w2v_model.corpus_count, epochs=20)
print(f"Time elapsed: {time.time() - start}")

w2v_model.save("custom_model/w2v_ir1.model")
