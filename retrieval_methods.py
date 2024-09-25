import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

class DenseRetrieval:
    def __init__(self, corpus):
        self.corpus = corpus
        self.index = faiss.IndexFlatL2(len(corpus[0]))
    
    def build_index(self):
        self.index.add(np.array(self.corpus).astype('float32'))
    
    def retrieve(self, query_vector, top_k=5):
        _, indices = self.index.search(np.array([query_vector]).astype('float32'), top_k)
        return indices

class SparseRetrieval:
    def __init__(self, corpus):
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query, top_k=5):
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        return np.argsort(doc_scores)[-top_k:]
