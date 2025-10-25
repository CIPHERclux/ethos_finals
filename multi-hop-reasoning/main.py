# File: main.py
"""
Main script to run inference on test data
"""
import sys
import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils.data_loader import DataLoader
from utils.embedding_generator import EmbeddingGenerator
from indexing.semantic_index import SemanticIndex
from inference.llm_client import LLMClient
from inference.reasoning_engine import ReasoningEngine
from config import config

def load_index_and_models():
    """Load pre-built index and models"""
    print("Loading semantic index and models...")
    
    # Load semantic index
    semantic_index = SemanticIndex(dimension=config.index.index_dimension)
    semantic_index.load(config.processing.index_dir)
    
    # Load embedding generator
    embedding_generator = EmbeddingGenerator(config.index.embedding_model)
    
    # Initialize LLM client
    llm_client = LLMClient()
    
    return semantic_index, embedding_generator, llm_client

def process_test_file(test_csv_path: str, output_csv_path: str):
    """
    Process test file and generate predictions
    """
    print("=" * 80)
    print("PHASE 2: ONLINE INFERENCE - Processing Test Data")
    print("=" * 80)
    
    # Load index and models
    semantic_index, embedding_generator, llm_client = load_index_and_models()
    
    # Initialize reasoning engine
    reasoning_engine = ReasoningEngine(
        semantic_index=semantic_index,
        embedding_generator=embedding_generator,
        llm_client=llm_client
    )
    
    # Load test data
    print(f"\nLoading test data from {test_csv_path}...")
    test_df = DataLoader.load_test_data(test_csv_path)
    print(f"Loaded {len(test_df)} test examples")
    
    if len(test_df) == 0:
        print("ERROR: No test examples loaded! Please check your test data file.")
        return None
    
    # Process each test example
    results = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
        question = row['question']
        context = row['parsed_context']
        question_type = row.get('type', 'bridge')  # Default to bridge if not specified
        level = row.get('level', 'medium')  # Default to medium if not specified
        
        # Get supporting facts hint if available
        supporting_facts_hint = None
        if 'parsed_supporting_facts' in row and len(row['parsed_supporting_facts']) > 0:
            # Extract unique document titles
            supporting_facts_hint = list(set(
                doc_title for doc_title, _ in row['parsed_supporting_facts']
            ))
        
        # Get answer
        try:
            result = reasoning_engine.answer_question(
                question=question,
                context=context,
                question_type=question_type,
                level=level,
                supporting_facts_hint=supporting_facts_hint
            )
            
            # Format supporting facts for output
            supporting_facts_formatted = {
                'title': [sf[0] for sf in result['supporting_facts']],
                'sent_id': [sf[1] for sf in result['supporting_facts']]
            }
            
            results.append({
                'id': row['id'],
                'question': question,
                'answer': result['answer'],
                'supporting_facts': str(supporting_facts_formatted),
                'reasoning': result['reasoning'],
                'confidence': result['confidence']
            })
            
        except Exception as e:
            print(f"\nError processing question {row['id']}: {e}")
            results.append({
                'id': row['id'],
                'question': question,
                'answer': "",
                'supporting_facts': "{'title': [], 'sent_id': []}",
                'reasoning': f"Error: {str(e)}",
                'confidence': 0.0
            })
    
    # Create output dataframe
    output_df = pd.DataFrame(results)
    
    # Save to CSV
    output_df.to_csv(output_csv_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to {output_csv_path}")
    print(f"{'='*80}")
    
    # Print statistics
    print("\nStatistics:")
    print(f"Total questions: {len(output_df)}")
    if 'answer' in output_df.columns and len(output_df) > 0:
        non_empty_answers = (output_df['answer'].notna() & (output_df['answer'] != '')).sum()
        print(f"Questions with answers: {non_empty_answers}")
    if 'confidence' in output_df.columns and len(output_df) > 0:
        print(f"Average confidence: {output_df['confidence'].mean():.2f}")
    
    return output_df

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Hop Reasoning System for Llama 3.1 8B"
    )
    parser.add_argument(
        '--mode',
        choices=['build', 'infer', 'both'],
        required=True,
        help='Mode: build index, run inference, or both'
    )
    parser.add_argument(
        '--train',
        type=str,
        help='Path to training CSV file (required for build mode)'
    )
    parser.add_argument(
        '--test',
        type=str,
        help='Path to test CSV file (required for infer mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to output CSV file'
    )
    
    args = parser.parse_args()
    
    if args.mode in ['build', 'both']:
        if not args.train:
            print("Error: --train argument required for build mode")
            sys.exit(1)
        
        from preprocessing.build_index import build_index_from_training_data
        build_index_from_training_data(args.train)
    
    if args.mode in ['infer', 'both']:
        if not args.test:
            print("Error: --test argument required for infer mode")
            sys.exit(1)
        
        result = process_test_file(args.test, args.output)
        if result is None:
            sys.exit(1)

if __name__ == "__main__":
    main()