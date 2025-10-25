"""Main script to build the semantic index from training data"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.data_loader import DataLoader
from utils.pattern_extractor import PatternExtractor
from utils.embedding_generator import EmbeddingGenerator
from indexing.semantic_index import SemanticIndex
from config import config
from tqdm import tqdm

def build_index_from_training_data(train_csv_path: str):
    print("=" * 80)
    print("PHASE 1: OFFLINE PREPROCESSING - Building Semantic Index")
    print("=" * 80)
    
    print("\n[1/5] Loading training data...")
    df = DataLoader.load_training_data(train_csv_path)
    print(f"Loaded {len(df)} valid training examples")
    
    if len(df) == 0:
        print("ERROR: No valid training data found!")
        sys.exit(1)
    
    print("\n[2/5] Extracting reasoning patterns...")
    pattern_extractor = PatternExtractor()
    patterns = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            if not isinstance(row.get('question'), str):
                continue
            
            pattern = pattern_extractor.extract_pattern(
                question_id=str(row['id']),
                question=row['question'],
                question_type=row.get('type', 'bridge'),
                level=row.get('level', 'medium'),
                supporting_facts=row['parsed_supporting_facts'],
                context=row['parsed_context'],
                answer=row.get('answer', '')
            )
            patterns.append(pattern)
        except:
            continue
    
    print(f"Extracted {len(patterns)} valid reasoning patterns")
    
    if len(patterns) == 0:
        print("ERROR: No valid patterns extracted!")
        sys.exit(1)
    
    print("\n[3/5] Generating question embeddings...")
    embedding_generator = EmbeddingGenerator(config.index.embedding_model)
    
    questions = [p.question for p in patterns]
    embeddings = embedding_generator.encode_questions(questions, batch_size=config.processing.batch_size)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    valid_patterns = []
    valid_embeddings = []
    for pattern, embedding in zip(patterns, embeddings):
        if not np.all(embedding == 0):
            valid_patterns.append(pattern)
            valid_embeddings.append(embedding)
    
    print(f"Valid patterns with embeddings: {len(valid_patterns)}")
    
    if len(valid_patterns) == 0:
        print("ERROR: No valid embeddings generated!")
        sys.exit(1)
    
    valid_embeddings = np.array(valid_embeddings)
    
    print("\n[4/5] Building semantic index...")
    semantic_index = SemanticIndex(dimension=config.index.index_dimension)
    semantic_index.build_index(valid_patterns, valid_embeddings)
    
    print("\n[5/5] Saving index to disk...")
    semantic_index.save(config.processing.index_dir)
    
    import pickle
    with open(os.path.join(config.processing.index_dir, "config.pkl"), 'wb') as f:
        pickle.dump({
            'embedding_model': config.index.embedding_model,
            'dimension': config.index.index_dimension
        }, f)
    
    print("\n" + "=" * 80)
    print("Index building complete!")
    print("=" * 80)
    
    return semantic_index, embedding_generator

if __name__ == "__main__":
    train_csv = "train.csv"
    
    if not os.path.exists(train_csv):
        print(f"Error: Training file {train_csv} not found!")
        sys.exit(1)
    
    build_index_from_training_data(train_csv)