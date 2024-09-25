from retrieval.retrieval_methods.py import DenseRetrieval, SparseRetrieval
from generation.generation_models import TextGenerator
import numpy as np

class RAGPipeline:
    def __init__(self, retrieval_type, generation_type, corpus):
        if retrieval_type == "dense":
            self.retrieval_model = DenseRetrieval(corpus)
        elif retrieval_type == "sparse":
            self.retrieval_model = SparseRetrieval(corpus)
        else:
            raise ValueError("Unknown retrieval type")
        
        self.generator = TextGenerator(generation_type)

    def retrieve_and_generate(self, query):
        if isinstance(self.retrieval_model, DenseRetrieval):
            query_vector = np.random.rand(len(self.retrieval_model.corpus[0]))  # Example vector
            retrieved_indices = self.retrieval_model.retrieve(query_vector)
        else:
            retrieved_indices = self.retrieval_model.retrieve(query)

        retrieved_docs = [self.retrieval_model.corpus[idx] for idx in retrieved_indices]
        combined_text = " ".join(retrieved_docs)
        generated_text = self.generator.generate_text(combined_text)
        
        return generated_text
