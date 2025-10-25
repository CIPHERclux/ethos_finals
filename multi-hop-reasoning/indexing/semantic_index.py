"""Build and query semantic index for fast example retrieval"""
import faiss
import numpy as np
from typing import List, Dict
import pickle
import os
from dataclasses import dataclass

@dataclass
class IndexedExample:
    id: str
    question: str
    question_type: str
    level: str
    pattern: object
    embedding: np.ndarray

class SemanticIndex:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.indices: Dict[str, Dict[str, object]] = {
            'bridge': {}, 'comparison': {}
        }
        self.indexed_examples: Dict[str, Dict[str, List[IndexedExample]]] = {
            'bridge': {'easy': [], 'medium': [], 'hard': []},
            'comparison': {'easy': [], 'medium': [], 'hard': []}
        }
    
    def build_index(self, patterns: List, embeddings: np.ndarray):
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))
        grouped = {
            'bridge': {'easy': [], 'medium': [], 'hard': []},
            'comparison': {'easy': [], 'medium': [], 'hard': []}
        }
        
        for pattern, embedding in zip(patterns, embeddings):
            embedding = np.ascontiguousarray(embedding.astype('float32'))
            example = IndexedExample(
                id=pattern.question_id,
                question=pattern.question,
                question_type=pattern.question_type,
                level=pattern.level,
                pattern=pattern,
                embedding=embedding
            )
            grouped[pattern.question_type][pattern.level].append(example)
        
        for q_type in ['bridge', 'comparison']:
            for level in ['easy', 'medium', 'hard']:
                examples = grouped[q_type][level]
                if len(examples) == 0:
                    continue
                
                self.indexed_examples[q_type][level] = examples
                embeddings_matrix = np.vstack([ex.embedding for ex in examples])
                embeddings_matrix = np.ascontiguousarray(embeddings_matrix.astype('float32'))
                
                index = faiss.IndexFlatIP(self.dimension)
                embeddings_normalized = embeddings_matrix.copy()
                faiss.normalize_L2(embeddings_normalized)
                index.add(embeddings_normalized)
                
                self.indices[q_type][level] = index
        
        print("Index building complete!")
    
    def search(self, query_embedding: np.ndarray, question_type: str, level: str, top_k: int = 5) -> List[IndexedExample]:
        if question_type not in self.indices or level not in self.indices[question_type]:
            return []
        
        index = self.indices[question_type].get(level)
        examples = self.indexed_examples[question_type][level]
        
        if index is None or len(examples) == 0:
            return []
        
        query = np.ascontiguousarray(query_embedding.astype('float32')).reshape(1, -1)
        faiss.normalize_L2(query)
        
        top_k = min(top_k, len(examples))
        distances, indices = index.search(query, top_k)
        
        return [examples[idx] for idx in indices[0] if 0 <= idx < len(examples)]
    
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for q_type in ['bridge', 'comparison']:
            for level in ['easy', 'medium', 'hard']:
                if level in self.indices[q_type] and self.indices[q_type][level] is not None:
                    faiss.write_index(self.indices[q_type][level], 
                                    os.path.join(directory, f"{q_type}_{level}.index"))
        
        with open(os.path.join(directory, "indexed_examples.pkl"), 'wb') as f:
            pickle.dump(self.indexed_examples, f)
        print(f"Index saved to {directory}")
    
    def load(self, directory: str):
        for q_type in ['bridge', 'comparison']:
            for level in ['easy', 'medium', 'hard']:
                index_path = os.path.join(directory, f"{q_type}_{level}.index")
                if os.path.exists(index_path):
                    self.indices[q_type][level] = faiss.read_index(index_path)
        
        with open(os.path.join(directory, "indexed_examples.pkl"), 'rb') as f:
            self.indexed_examples = pickle.load(f)
        print(f"Index loaded from {directory}")