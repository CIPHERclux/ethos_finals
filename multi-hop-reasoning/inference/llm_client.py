# File: inference/llm_client.py
"""
Client for interacting with Groq API
"""
import os
from groq import Groq
from typing import Dict, Optional, List
from config import config

class LLMClient:
    """Client for Groq API with Llama 3.1 8B"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.model_name = config.model.model_name
    
    def generate(
        self,
        prompt: str,
        generation_params: Dict,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate response from LLM
        """
        messages = []
        
        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **generation_params
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def generate_with_self_consistency(
        self,
        prompt: str,
        generation_params: Dict,
        num_samples: int = 5,
        system_message: Optional[str] = None
    ) -> List[str]:
        """
        Generate multiple responses for self-consistency
        """
        responses = []
        
        for _ in range(num_samples):
            response = self.generate(prompt, generation_params, system_message)
            if response:
                responses.append(response)
        
        return responses