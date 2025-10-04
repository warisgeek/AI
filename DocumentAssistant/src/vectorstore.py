import faiss
import numpy as np

class FaissStore:
    def __init__(self, dim:int=384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.docs = []

    def add_documents(self, docs: list[str], vectors: np.ndarray):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self.index.add(vectors.astype('float32'))
        self.docs.extend(docs)

    def similarity_search(self, query_vector: np.ndarray, k:int=4):
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        D, I = self.index.search(query_vector.astype('float32'), k)
        results = []
        for idx in I[0]:
            if idx < len(self.docs):
                results.append(self.docs[idx])
        return results
