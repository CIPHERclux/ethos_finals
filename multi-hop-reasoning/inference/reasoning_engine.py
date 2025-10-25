# File: inference/reasoning_engine.py
"""
Main reasoning engine that orchestrates the inference process
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple
from utils.embedding_generator import EmbeddingGenerator
from indexing.semantic_index import SemanticIndex
from prompts.prompt_builder import PromptBuilder
from inference.llm_client import LLMClient
from inference.response_parser import ResponseParser
from config import config

class ReasoningEngine:
    """Main engine for multi-hop reasoning"""
    
    def __init__(
        self,
        semantic_index: SemanticIndex,
        embedding_generator: EmbeddingGenerator,
        llm_client: LLMClient
    ):
        self.semantic_index = semantic_index
        self.embedding_generator = embedding_generator
        self.llm_client = llm_client
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
    
    def answer_question(
        self,
        question: str,
        context: Dict[str, List[str]],
        question_type: str,
        level: str,
        supporting_facts_hint: Optional[List[str]] = None
    ) -> Dict:
        """
        Answer a single question using multi-hop reasoning
        
        Returns:
            Dict with keys: 'answer', 'reasoning', 'supporting_facts', 'confidence'
        """
        # Validate inputs
        if not isinstance(question, str) or len(question.strip()) == 0:
            return {
                'answer': "",
                'reasoning': "Invalid question",
                'supporting_facts': [],
                'confidence': 0.0
            }
        
        if not context or len(context) == 0:
            return {
                'answer': "",
                'reasoning': "No context provided",
                'supporting_facts': [],
                'confidence': 0.0
            }
        
        print(f"\nProcessing question: {question[:100]}...")
        
        # Step 1: Determine strategy based on difficulty
        use_self_consistency = (
            level == 'hard' and config.use_self_consistency
        )
        
        # Step 2: Get generation parameters
        gen_params = self._get_generation_params(level)
        
        # Step 3: Retrieve similar examples
        similar_examples = self._retrieve_similar_examples(
            question, question_type, level
        )
        
        # Step 4: Build prompt
        prompt = self.prompt_builder.build_prompt(
            question=question,
            context=context,
            question_type=question_type,
            level=level,
            similar_examples=similar_examples,
            supporting_facts_hint=supporting_facts_hint
        )
        
        # Step 5: Generate response(s)
        try:
            if use_self_consistency:
                responses = self.llm_client.generate_with_self_consistency(
                    prompt=prompt,
                    generation_params=gen_params,
                    num_samples=config.self_consistency_samples
                )
                
                # Majority vote
                answer = self.response_parser.majority_vote(responses)
                reasoning = responses[0] if responses else ""
                
            else:
                response = self.llm_client.generate(prompt, gen_params)
                answer = self.response_parser.extract_answer(response)
                reasoning = response
        except Exception as e:
            print(f"Error during generation: {e}")
            return {
                'answer': "",
                'reasoning': f"Error: {str(e)}",
                'supporting_facts': [],
                'confidence': 0.0
            }
        
        # Step 6: Extract supporting facts
        supporting_facts = self.response_parser.extract_supporting_facts(reasoning)
        
        # Step 7: Calculate confidence
        confidence = self.response_parser.calculate_confidence(
            reasoning,
            expected_hops=2 if question_type == 'bridge' else 2
        )
        
        return {
            'answer': answer or "",
            'reasoning': reasoning,
            'supporting_facts': supporting_facts,
            'confidence': confidence
        }
    
    def _get_generation_params(self, level: str) -> Dict:
        """Get generation parameters based on difficulty level"""
        if level == 'easy':
            return config.model.easy_params
        elif level == 'medium':
            return config.model.medium_params
        else:
            return config.model.hard_params
    
    def _retrieve_similar_examples(
        self,
        question: str,
        question_type: str,
        level: str
    ) -> List:
        """Retrieve similar examples from semantic index"""
        
        # Determine number of examples based on level
        if level == 'easy':
            k = config.easy_num_examples
        elif level == 'medium':
            k = config.medium_num_examples
        else:
            k = config.hard_num_examples
        
        # Generate query embedding
        query_embedding = self.embedding_generator.encode_single(question)
        
        # Check if embedding is valid
        import numpy as np
        if np.all(query_embedding == 0):
            print("Warning: Zero embedding generated for question")
            return []
        
        # Search index
        similar_examples = self.semantic_index.search(
            query_embedding=query_embedding,
            question_type=question_type,
            level=level,
            top_k=k
        )
        
        return similar_examples