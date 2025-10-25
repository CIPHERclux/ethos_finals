# File: utils/data_loader.py
"""
Data loading and preprocessing utilities
"""
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
import ast
import numpy as np

class DataLoader:
    """Load and preprocess HotpotQA data"""
    
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load CSV file"""
        return pd.read_csv(filepath)
    
    @staticmethod
    def parse_supporting_facts(sf_str: str) -> List[Tuple[str, int]]:
        """
        Parse supporting facts from string format
        Format: "{'title': ['doc1', 'doc2'], 'sent_id': [0, 1]}"
        """
        try:
            # Handle NaN/None
            if pd.isna(sf_str) or sf_str is None:
                return []
            
            # Handle string representation of dict
            if isinstance(sf_str, str):
                sf_dict = ast.literal_eval(sf_str)
            else:
                sf_dict = sf_str
            
            titles = sf_dict.get('title', [])
            sent_ids = sf_dict.get('sent_id', [])
            
            return list(zip(titles, sent_ids))
        except Exception as e:
            # Silently return empty list for invalid data
            return []
    
    @staticmethod
    def parse_context(context_str: str) -> Dict[str, List[str]]:
        """
        Parse context from string format - FIXED VERSION
        """
        try:
            # Handle NaN/None/float
            if pd.isna(context_str) or context_str is None or isinstance(context_str, float):
                return {}
            
            # Convert string to dict
            if isinstance(context_str, str):
                context_dict = ast.literal_eval(context_str)
            else:
                context_dict = context_str
            
            # Check if it's already in the right format
            if isinstance(context_dict, dict) and all(isinstance(v, list) for v in context_dict.values()):
                # Already in format {title: [sentences]}
                return context_dict
            
            # Otherwise, restructure from {title: [...], sentences: [[...]]}
            result = {}
            titles = context_dict.get('title', [])
            sentences_list = context_dict.get('sentences', [])
            
            # Convert numpy arrays to lists if needed
            if hasattr(titles, 'tolist'):
                titles = titles.tolist()
            if hasattr(sentences_list, 'tolist'):
                sentences_list = sentences_list.tolist()
            
            for title, sentences in zip(titles, sentences_list):
                # Convert sentences to list if it's a numpy array
                if hasattr(sentences, 'tolist'):
                    sentences = sentences.tolist()
                # Convert each sentence to string
                sentences = [str(s) for s in sentences]
                result[str(title)] = sentences
            
            return result
        except Exception as e:
            print(f"Error parsing context: {e}")
            print(f"Context string: {str(context_str)[:200]}...")
            return {}
    
    @staticmethod
    def load_training_data(filepath: str) -> pd.DataFrame:
        """Load and parse training data"""
        df = DataLoader.load_csv(filepath)
        
        # Drop rows with missing critical data
        print(f"Original rows: {len(df)}")
        
        # Drop rows where question or context is NaN
        df = df.dropna(subset=['question', 'context'])
        print(f"After dropping NaN questions/context: {len(df)}")
        
        # Parse supporting facts
        if 'supporting_facts' in df.columns:
            df['parsed_supporting_facts'] = df['supporting_facts'].apply(
                DataLoader.parse_supporting_facts
            )
        else:
            df['parsed_supporting_facts'] = [[] for _ in range(len(df))]
        
        # Parse context
        df['parsed_context'] = df['context'].apply(
            DataLoader.parse_context
        )
        
        # Drop rows where context parsing failed (empty dict)
        df = df[df['parsed_context'].apply(lambda x: isinstance(x, dict) and len(x) > 0)]
        print(f"After dropping invalid contexts: {len(df)}")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    @staticmethod
    def load_test_data(filepath: str) -> pd.DataFrame:
        """Load and parse test data (may not have answers or supporting facts)"""
        df = DataLoader.load_csv(filepath)
        
        # Drop rows with missing critical data
        print(f"Original test rows: {len(df)}")
        
        # Drop rows where question or context is NaN
        df = df.dropna(subset=['question', 'context'])
        print(f"After dropping NaN questions/context: {len(df)}")
        
        # Parse context
        print("Parsing contexts...")
        df['parsed_context'] = df['context'].apply(
            DataLoader.parse_context
        )
        
        # Check how many failed
        valid_contexts = df['parsed_context'].apply(lambda x: isinstance(x, dict) and len(x) > 0)
        print(f"Valid contexts: {valid_contexts.sum()} / {len(df)}")
        
        # Drop rows where context parsing failed
        df = df[valid_contexts]
        print(f"After dropping invalid contexts: {len(df)}")
        
        # Parse supporting facts if present
        if 'supporting_facts' in df.columns:
            df['parsed_supporting_facts'] = df['supporting_facts'].apply(
                DataLoader.parse_supporting_facts
            )
        else:
            df['parsed_supporting_facts'] = [[] for _ in range(len(df))]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df