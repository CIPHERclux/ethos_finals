"""
Program-Aided Language (PAL) solver.
Contract: Generate Python code, execute in sandbox, return single value or failure.
"""

import re
import os
import time
from typing import Dict, Any, List
from groq import Groq

# Get API key from environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


class PALSolver:
    def __init__(self, timeout: int = 5, max_retries: int = 2, temperature: float = 0.0):
        """
        Initialize PAL solver.
        
        Args:
            timeout: Maximum execution time for generated code
            max_retries: Number of retry attempts on failure
            temperature: Sampling temperature (0.0 = deterministic)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.client = Groq(api_key=GROQ_API_KEY)

    def solve(self, question: str, few_shots: List[str] = None) -> Dict[str, Any]:
        """
        Solve a question by generating and executing Python code.
        
        Args:
            question: Math problem to solve
            few_shots: Optional list of example solutions
            
        Returns:
            Dictionary with success status, code, result, and optional error
        """
        
        for attempt in range(self.max_retries):
            try:
                # Generate code
                raw_code = self._generate_code(question, few_shots)
                
                # Clean code (remove markdown)
                code = self._extract_code(raw_code)
                
                # Validate code has 'answer' variable
                if not code or 'answer' not in code:
                    continue
                
                # Execute code
                result = self._execute_code(code)
                
                return {
                    "success": True,
                    "code": code,
                    "result": result
                }
                
            except Exception as e:
                # Continue to next attempt
                if attempt < self.max_retries - 1:
                    time.sleep(0.5)  # Small delay between retries
                continue
        
        # All attempts failed
        return {
            "success": False,
            "code": None,
            "result": None,
            "error": "All attempts failed"
        }

    def _generate_code(self, question: str, few_shots: List[str] = None) -> str:
        """
        Generate Python code to solve the question with rate limit handling.
        
        Args:
            question: Math problem to solve
            few_shots: Optional list of example solutions
            
        Returns:
            String containing Python code
        """
        # Build few-shot context (limit to 2 examples)
        few_shot_text = ""
        if few_shots:
            few_shot_text = "\n\n".join(few_shots[:2])
        
        # Super explicit prompt
        system_prompt = """You are a Python code generator for solving math problems.

RULES:
1. Output ONLY executable Python code
2. Do NOT use markdown code blocks (no ```)
3. Use only built-in Python (no imports)
4. Assign the final numeric answer to a variable named `answer`
5. Use descriptive variable names

Example:
distance_ab = 100
distance_bc = distance_ab + 50
distance_cd = distance_bc * 2
total_distance = distance_ab + distance_bc + distance_cd
answer = total_distance"""

        user_prompt = f"""{few_shot_text}

Problem: {question}

Write Python code that solves this step-by-step. Output ONLY the code."""

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
                    max_tokens=512,
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limit errors
                if "rate_limit" in error_msg or "429" in error_msg:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"  ⏳ Rate limit hit (PAL attempt {attempt+1}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                    if attempt == max_retries - 1:
                        raise Exception("Rate limit exceeded after retries")
                else:
                    # Other error - raise immediately
                    raise e
        
        raise Exception("Max retries exceeded")

    def _extract_code(self, text: str) -> str:
        """
        Extract Python code from LLM response.
        Handles markdown blocks and other formatting.
        
        Args:
            text: Raw LLM response
            
        Returns:
            Clean Python code
        """
        code = text.strip()
        
        # Remove markdown code blocks - all patterns
        code = re.sub(r'^```python\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```', '', code)
        code = re.sub(r'```$', '', code)
        
        # Remove explanatory prefixes
        code = re.sub(r'^(Here is|Here\'s|Solution:).*?\n', '', code, flags=re.IGNORECASE)
        
        return code.strip()

    def _execute_code(self, code: str) -> Any:
        """
        Execute Python code in a restricted, safe environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            Value of the `answer` variable after execution
        """
        # Create safe namespace with only basic built-ins
        safe_builtins = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'range': range,
            'pow': pow,
            'divmod': divmod,
        }
        
        namespace = {'__builtins__': safe_builtins}
        
        # Execute code
        try:
            exec(code, namespace)
        except Exception as e:
            raise RuntimeError(f"Code execution failed: {type(e).__name__}: {e}")
        
        # Extract answer
        if 'answer' not in namespace:
            raise ValueError("Code executed but 'answer' variable not found")
        
        result = namespace['answer']
        
        if result is None:
            raise ValueError("Answer variable is None")
        
        return result