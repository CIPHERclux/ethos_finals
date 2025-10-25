"""
Verifier to reconcile PAL and CoT results.
Contract: Apply rules to choose between methods, mark confidence.
"""
import re
from typing import Dict, Any


class Verifier:
    def __init__(self, prefer_pal_for_arithmetic: bool = True):
        self.prefer_pal = prefer_pal_for_arithmetic
    
    def verify(self, pal_result: Dict[str, Any], 
           cot_result: Dict[str, Any],
           question: str) -> Dict[str, Any]:
        """
        Verify and reconcile PAL and CoT results.
        """
        pal_success = pal_result.get("success", False)
        cot_success = cot_result.get("result") is not None
        
        pal_answer = pal_result.get("result")
        cot_answer = cot_result.get("result")
        cot_confidence = cot_result.get("confidence", 0.0)
        
        # CASE 1: Both succeeded
        if pal_success and cot_success:
            pal_norm = self._normalize_answer(pal_answer)
            cot_norm = self._normalize_answer(cot_answer)
            
            if pal_norm == cot_norm:
                # They agree!
                return {
                    "final_answer": str(pal_answer),
                    "method": "both_agree",
                    "confidence": 1.0
                }
            
            # They disagree - MORE SOPHISTICATED LOGIC
            is_arithmetic = self._is_arithmetic_question(question)
            is_complex_word_problem = self._is_complex_word_problem(question)
            
            # NEW: For complex word problems, trust high-confidence CoT
            if is_complex_word_problem and cot_confidence >= 0.65:
                return {
                    "final_answer": str(cot_answer),
                    "method": "cot_complex_problem",
                    "confidence": cot_confidence
                }
            
            # For simple arithmetic, prefer PAL
            if is_arithmetic and not is_complex_word_problem and self.prefer_pal:
                return {
                    "final_answer": str(pal_answer),
                    "method": "pal_simple_arithmetic",
                    "confidence": 0.75
                }
            
            # High CoT confidence (70%+) wins
            if cot_confidence >= 0.7:
                return {
                    "final_answer": str(cot_answer),
                    "method": "cot_high_confidence",
                    "confidence": cot_confidence
                }
            
            # NEW: Medium CoT confidence (60-70%) vs PAL - use CoT
            if cot_confidence >= 0.6:
                return {
                    "final_answer": str(cot_answer),
                    "method": "cot_medium_confidence",
                    "confidence": cot_confidence
                }
            
            # Default to PAL
            return {
                "final_answer": str(pal_answer),
                "method": "pal_default",
                "confidence": 0.65
            }
    
        # CASE 2: Only PAL succeeded
        if pal_success:
            return {
                "final_answer": str(pal_answer),
                "method": "pal_only",
                "confidence": 0.7
            }
        
        # CASE 3: Only CoT succeeded
        if cot_success:
            return {
                "final_answer": str(cot_answer),
                "method": "cot_only",
                "confidence": max(0.6, cot_confidence)
            }
        
        # CASE 4: Both failed
        return {
            "final_answer": "0",
            "method": "fallback",
            "confidence": 0.0
        }

    def _is_complex_word_problem(self, question: str) -> bool:
        """Detect complex multi-step word problems."""
        complexity_indicators = [
            'every', 'each', 'doubles', 'triples',
            'weekend', 'weekday', 'except',
            'both', 'either', 'between',
            'saturday', 'sunday', 'monday',
            'morning', 'evening', 'afternoon'
        ]
        
        q_lower = question.lower()
        matches = sum(1 for indicator in complexity_indicators if indicator in q_lower)
        
        # If 2+ complexity indicators, it's complex
        return matches >= 2
        
    def _normalize_answer(self, answer: Any) -> str:
        """Normalize answer for comparison."""
        s = str(answer).strip().lower()
        # Remove common formatting
        s = s.replace(',', '').replace('$', '').replace('%', '')
        s = re.sub(r'\s+', '', s)
        return s
    
    def _is_arithmetic_question(self, question: str) -> bool:
        """Detect if question is primarily arithmetic."""
        arithmetic_indicators = [
            'calculate', 'compute', 'how many', 'how much',
            'total', 'sum', 'difference', 'product', 'quotient',
            '+', '-', '*', '/', 'divide', 'multiply',
            'cost', 'price', 'earn', 'spend', 'pay'
        ]
        
        q_lower = question.lower()
        return any(indicator in q_lower for indicator in arithmetic_indicators)