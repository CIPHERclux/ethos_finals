"""Generate embeddings for questions and build semantic index"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode_questions(self, questions: List[str], batch_size: int = 32) -> np.ndarray:
        valid_questions = []
        valid_indices = []
        
        for idx, q in enumerate(questions):
            if isinstance(q, str) and len(q.strip()) > 0:
                valid_questions.append(q)
                valid_indices.append(idx)
        
        print(f"Encoding {len(valid_questions)} valid questions out of {len(questions)} total")
        
        if len(valid_questions) == 0:
            return np.array([]).astype('float32')
        
        embeddings = self.model.encode(
            valid_questions,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype('float32')
        
        full_embeddings = np.zeros((len(questions), self.dimension), dtype='float32')
        full_embeddings[valid_indices] = embeddings
        
        return full_embeddings
    
    def encode_single(self, question: str) -> np.ndarray:
        if not isinstance(question, str) or len(question.strip()) == 0:
            return np.zeros(self.dimension, dtype='float32')
        return self.model.encode(question, convert_to_numpy=True).astype('float32')
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        np.save(filepath, embeddings.astype('float32'))
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        return np.load(filepath).astype('float32')