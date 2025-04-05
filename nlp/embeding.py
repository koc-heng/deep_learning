import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import random

# 讀格式
data = pd.read_csv('tokenized_data.csv')
data = data.dropna(subset=['tokenized_title'])

documents = []
for i, row in data.iterrows():
    tokens = row['tokenized_title'].split()
    documents.append(TaggedDocument(words=tokens, tags=[i]))

total_docs = len(documents)
print("total：", total_docs)  


# traing
model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=8, epochs=30)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)


# 計算餘弦相似度
def cosine_similarity(vec_a, vec_b):
    numerator = np.dot(vec_a, vec_b)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0
    return numerator / denom

# 4. leave-one 評估
def evaluate_leave_one(model, eval_docs):
    self_sim_count = 0
    second_self_sim_count = 0
    total = len(eval_docs)
    
    for idx, doc in enumerate(eval_docs):
        inferred_vec = model.infer_vector(doc.words)
        similarities = []
        for j in range(total):
            sim = cosine_similarity(inferred_vec, model.dv[eval_docs[j].tags[0]])
            similarities.append((j, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        top1 = similarities[0][0]
        top2 = similarities[1][0]
        if top1 == idx:
            self_sim_count += 1
        if top1 == idx or top2 == idx:
            second_self_sim_count += 1
    return self_sim_count / total, second_self_sim_count / total


# 評估 1000 筆資料
eval_indices = random.sample(range(total_docs), 1000)
eval_docs = [documents[i] for i in eval_indices]

model.save("my_doc2vec.model")
print("model save to my_doc2vec.model")

self_sim, second_self_sim = evaluate_leave_one(model, eval_docs)
print("Self Similarity:", self_sim)
print("Second Self Similarity:", second_self_sim)