"""
Load CSV data and return list of questions or training tuples.
Contract: Preserve row order for aligned output writing.
"""
import pandas as pd
from typing import List, Tuple
import re


def load_test_data(filepath: str) -> List[str]:
    """
    Load test data from CSV.
    Input: Path to testmath.csv with 'question' column
    Output: List[str] of questions in original order
    """
    df = pd.read_csv(filepath)
    # Drop any NaN values
    df = df.dropna(subset=['question'])
    # Convert to string to handle any numeric questions
    questions = df['question'].astype(str).tolist()
    return questions


def load_training_data(filepath: str) -> List[Tuple[str, str]]:
    """
    Load training data from CSV.
    Input: Path to training CSV with 'question' and 'answer' columns
    Output: List of (question, answer) tuples
    """
    df = pd.read_csv(filepath)
    # Drop rows with NaN in question or answer
    df = df.dropna(subset=['question', 'answer'])
    # Convert to string
    df['question'] = df['question'].astype(str)
    df['answer'] = df['answer'].astype(str)
    data = list(zip(df['question'].tolist(), df['answer'].tolist()))
    return data


def extract_final_answer(answer_text: str) -> str:
    """
    Extract the final answer from training format.
    Input: Answer string with "#### <value>" terminator
    Output: Clean final value
    """
    match = re.search(r'####\s*(.+?)(?:\n|$)', str(answer_text), re.MULTILINE)
    if match:
        return match.group(1).strip()
    return str(answer_text).strip()