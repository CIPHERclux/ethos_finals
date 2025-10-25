"""
Chain-of-Thought (CoT) solver with self-consistency.
Contract: Generate K reasoning chains, extract "#### ", return majority vote.
"""

import re
import os
import time
from typing import List, Dict, Any, Optional
from collections import Counter
from groq import Groq

# Get API key from environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


class CoTSolver:
    def __init__(self, num_samples: int = 3, temperature: float = 0.7):
        """
        Initialize CoT solver with self-consistency.
        
        Args:
            num_samples: Number of reasoning chains to generate
            temperature: Sampling temperature for diversity (0.7 recommended)
        """
        self.num_samples = num_samples
        self.temperature = temperature
        self.client = Groq(api_key=GROQ_API_KEY)

    def solve(self, question: str, few_shots: List[str] = None) -> Dict[str, Any]:
        """
        Solve a question using self-consistency across multiple reasoning chains.
        
        Args:
            question: Math problem to solve
            few_shots: Optional list of example solutions
            
        Returns:
            Dictionary with traces, extracted answers, result, and confidence
        """
        traces = []
        extracted_answers = []
        
        # Generate multiple reasoning chains
        for i in range(self.num_samples):
            try:
                trace = self._generate_reasoning(question, few_shots)
                traces.append(trace)
                
                # Extract final answer from each trace
                answer = self._extract_final_answer(trace)
                if answer:
                    extracted_answers.append(answer)
            except Exception as e:
                # Continue with other chains
                print(f"  ⚠️ CoT sample {i+1} failed: {e}")
                continue
        
        # Apply majority voting
        if extracted_answers:
            answer_counts = Counter(extracted_answers)
            majority_answer = answer_counts.most_common(1)[0][0]
            confidence = answer_counts[majority_answer] / len(extracted_answers)
            
            return {
                "traces": traces,
                "extracted_answers": extracted_answers,
                "result": majority_answer,
                "confidence": confidence
            }
        
        # No valid answers extracted
        return {
            "traces": traces,
            "extracted_answers": [],
            "result": None,
            "confidence": 0.0
        }

    def _generate_reasoning(self, question: str, few_shots: List[str] = None) -> str:
        """
        Generate a single reasoning chain for the question with rate limit handling.
        
        Args:
            question: Math problem to solve
            few_shots: Optional list of example solutions
            
        Returns:
            String containing reasoning chain ending with "#### [answer]"
        """
        # Build few-shot context (limit to 2 examples)
        few_shot_text = ""
        if few_shots:
            few_shot_text = "\n\n".join(few_shots[:2])
        
        system_prompt = """You are a mathematical reasoning assistant that solves problems step by step.

CRITICAL: Always end your solution with a line in this exact format:
#### [final_answer]

Where [final_answer] is just the numeric or text answer.

Example:
Problem: John has 5 apples and buys 3 more. How many total?

Solution:
Starting apples: 5
Apples bought: 3
Total = 5 + 3 = 8

#### 8"""

        user_prompt = f"""{few_shot_text}

Problem: {question}

Solve this step by step. Show all your work. End with: #### [answer]"""

        # Call Groq API with retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add delay to avoid rate limiting
                if attempt > 0:
                    time.sleep(1)
                
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=1024,
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limit errors
                if "rate_limit" in error_msg or "429" in error_msg:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"  ⏳ Rate limit hit (CoT attempt {attempt+1}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                    if attempt == max_retries - 1:
                        raise Exception("Rate limit exceeded after retries")
                else:
                    # Other error - raise immediately
                    raise e
        
        raise Exception("Max retries exceeded")

    def _extract_final_answer(self, trace: str) -> Optional[str]:
        """
        Extract the final answer from a reasoning trace.
        
        Args:
            trace: Reasoning chain containing "#### [answer]"
            
        Returns:
            Extracted answer string or None if not found
        """
        # Pattern 1: Match "#### " followed by the answer
        match = re.search(r'####\s*(.+?)(?:\n|$)', trace, re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            # Clean up common artifacts
            answer = re.sub(r'[*`\[\]]', '', answer).strip()
            return answer
        
        # Pattern 2: Fallback - look for "answer:" or "final answer:"
        match = re.search(r'(?:final\s+)?answer:?\s*(.+?)(?:\n|$)', trace, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'[*`\[\]]', '', answer).strip()
            return answer
        
        return None