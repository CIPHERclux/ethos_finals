"""
Few-shot retriever using embedding similarity.
Contract: Use cached embeddings, return similar problems with "#### <final>" format.
"""
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class FewShotRetriever:
    def __init__(self, training_data: List[Tuple[str, str]], 
                 cache_path: str, 
                 model_name: str = "all-MiniLM-L6-v2"):
        self.training_data = training_data
        self.cache_path = cache_path
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_or_compute_embeddings()

    def _load_or_compute_embeddings(self) -> np.ndarray:
        """Load cached embeddings or compute and save them."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cached = pickle.load(f)
                    
                    # STRICT VALIDATION: Must be proper numpy array
                    if isinstance(cached, np.ndarray):
                        # Check it's not an object array (which can contain dicts)
                        if cached.dtype == object:
                            print(f"⚠️  Cache contains object dtype, recomputing...")
                            os.remove(self.cache_path)
                        # Check dimensions
                        elif cached.ndim == 2 and cached.shape[0] == len(self.training_data):
                            print(f"✓ Loaded cached embeddings: {cached.shape}")
                            return cached
                        elif cached.ndim == 1:
                            print(f"⚠️  1D cache detected, recomputing...")
                            os.remove(self.cache_path)
                        else:
                            print(f"⚠️  Invalid cache shape {cached.shape}, expected ({len(self.training_data)}, ?)")
                            os.remove(self.cache_path)
                    else:
                        print(f"⚠️  Cache is not numpy array: {type(cached)}, recomputing...")
                        os.remove(self.cache_path)
                    
            except Exception as e:
                print(f"⚠️  Error loading cache: {e}, recomputing...")
                if os.path.exists(self.cache_path):
                    os.remove(self.cache_path)

        # Compute embeddings
        return self._compute_embeddings()
    
    def _compute_embeddings(self) -> np.ndarray:
        """Compute embeddings for all training questions."""
        questions = [str(q) for q, _ in self.training_data]
        print(f"Computing embeddings for {len(questions)} questions...")
        
        embeddings = self.model.encode(
            questions, 
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        
        # Force proper numpy array with float dtype
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Ensure 2D shape
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)
        
        # Validate
        assert embeddings.shape[0] == len(questions), f"Shape mismatch: {embeddings.shape[0]} != {len(questions)}"
        assert embeddings.dtype in [np.float32, np.float64], f"Invalid dtype: {embeddings.dtype}"
        
        # Save to cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(embeddings, f, protocol=4)
        
        print(f"✓ Cached embeddings - shape: {embeddings.shape}, dtype: {embeddings.dtype}")

        return embeddings

    def retrieve(self, question: str, k: int = 3) -> List[str]:
        """
        Retrieve K similar training examples.
        Output: List[str] of K training examples with "Question: ... Answer: ..." format.
        """
        # Encode query
        query_embedding = self.model.encode(
            [str(question)], 
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        
        # Ensure proper shape and dtype
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Validate embeddings before similarity computation
        if not isinstance(self.embeddings, np.ndarray):
            print(f"⚠️  Embeddings corrupted (type: {type(self.embeddings)}), recomputing...")
            self.embeddings = self._compute_embeddings()
        
        if self.embeddings.dtype == object:
            print(f"⚠️  Embeddings have object dtype, recomputing...")
            self.embeddings = self._compute_embeddings()
        
        # Compute cosine similarity
        try:
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        except Exception as e:
            print(f"⚠️  Similarity failed: {e}")
            print(f"Query: {query_embedding.shape}, {query_embedding.dtype}")
            print(f"Embeddings: {self.embeddings.shape}, {self.embeddings.dtype}")
            # Force recompute and retry
            self.embeddings = self._compute_embeddings()
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top K
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        examples = []
        for idx in top_k_indices:
            q, a = self.training_data[idx]
            examples.append(f"Question: {q}\nAnswer: {a}")

        return examples