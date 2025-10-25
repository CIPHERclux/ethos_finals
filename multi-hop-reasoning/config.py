# File: config.py
"""
Configuration settings for the multi-hop reasoning system
"""
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelConfig:
    """LLM model configuration"""
    model_name: str = "llama-3.1-8b-instant"
    
    # Generation parameters for different difficulty levels
    easy_params: Dict = None
    medium_params: Dict = None
    hard_params: Dict = None
    
    def __post_init__(self):
        self.easy_params = {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 800,
            "stop": ["FINAL ANSWER:"]
        }
        
        self.medium_params = {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 1000,
            "stop": ["FINAL ANSWER:"]
        }
        
        self.hard_params = {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 1200,
            "stop": ["FINAL ANSWER:"]
        }

@dataclass
class IndexConfig:
    """Vector index configuration"""
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    index_dimension: int = 768
    top_k_similar: int = 5
    
@dataclass
class ProcessingConfig:
    """Data processing configuration"""
    batch_size: int = 32
    num_workers: int = 4
    cache_dir: str = "./cache"
    index_dir: str = "./indices"
    
@dataclass
class SystemConfig:
    """Main system configuration"""
    model: ModelConfig = None
    index: IndexConfig = None
    processing: ProcessingConfig = None
    
    # Number of examples for few-shot
    easy_num_examples: int = 2
    medium_num_examples: int = 3
    hard_num_examples: int = 3
    
    # Self-consistency for hard questions
    use_self_consistency: bool = True
    self_consistency_samples: int = 5
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.index is None:
            self.index = IndexConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
            
        # Create directories
        os.makedirs(self.processing.cache_dir, exist_ok=True)
        os.makedirs(self.processing.index_dir, exist_ok=True)

# Global config instance
config = SystemConfig()