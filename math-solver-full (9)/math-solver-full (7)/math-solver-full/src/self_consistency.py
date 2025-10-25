"""
Self-consistency voting mechanism.
Contract: Aggregate multiple reasoning paths and return majority answer.
"""
from collections import Counter
from typing import List, Tuple

def vote_answers(answers: List[str]) -> Tuple[str, float]:
    if not answers:
        return None, 0.0
    normalized = [a.strip().lower() for a in answers if a]
    if not normalized:
        return None, 0.0
    counts = Counter(normalized)
    majority_answer, count = counts.most_common(1)[0]
    confidence = count / len(normalized)
    for orig in answers:
        if orig.strip().lower() == majority_answer:
            return orig.strip(), confidence
    return majority_answer, confidence
