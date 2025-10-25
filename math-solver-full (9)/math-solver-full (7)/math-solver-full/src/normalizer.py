"""
Answer normalizer.
Contract: Format numbers (int/float/currency), return clean string.
"""
import re


def normalize_answer(answer: str, question: str) -> str:
    """
    Normalize answer to proper format based on question context.
    
    Args:
        answer: Raw answer string
        question: Original question
    
    Returns:
        Normalized answer string
    """
    answer = str(answer).strip()
    
    # Remove markdown artifacts
    answer = re.sub(r'####\s*', '', answer)
    answer = re.sub(r'[*`\[\]]', '', answer)
    answer = answer.replace('<<', '').replace('>>', '')
    
    question_lower = question.lower()
    
    try:
        # Try to parse as number
        num_str = answer.replace('$', '').replace(',', '').strip()
        
        if '.' in num_str or 'e' in num_str.lower():
            num = float(num_str)
        else:
            num = int(float(num_str))
        
        # Format based on question type
        if 'how many' in question_lower:
            return str(int(num))
        elif '$' in question or 'dollar' in question_lower or 'cost' in question_lower or 'price' in question_lower:
            return f"{num:.2f}"
        else:
            # Integer if whole, else float
            if isinstance(num, float) and num.is_integer():
                return str(int(num))
            return str(num)
            
    except (ValueError, TypeError):
        # Not a number - return cleaned string
        return answer.strip()