# File: inference/response_parser.py
"""
Parse and extract answers from LLM responses
"""
import re
from typing import Optional, Dict, List
from collections import Counter

class ResponseParser:
    """Parse structured responses from LLM"""
    
    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract final answer from response
        """
        # Look for "FINAL ANSWER:" marker
        pattern = r"FINAL ANSWER:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, response, re.IGNORECASE)
        
        if match:
            answer = match.group(1).strip()
            return answer
        
        # Fallback: look for last line
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('Step'):
                return line
        
        return None
    
    def extract_reasoning_steps(self, response: str) -> Dict[str, str]:
        """
        Extract structured reasoning steps
        """
        steps = {}
        
        # Extract each hop
        hop_pattern = r"HOP (\d+):(.*?)(?=HOP \d+:|Step \d+:|FINAL ANSWER:|$)"
        hop_matches = re.finditer(hop_pattern, response, re.DOTALL | re.IGNORECASE)
        
        for match in hop_matches:
            hop_num = match.group(1)
            hop_content = match.group(2).strip()
            steps[f"hop_{hop_num}"] = hop_content
        
        # Extract answer synthesis
        synthesis_pattern = r"Step \d+ - Answer Synthesis:(.*?)(?=FINAL ANSWER:|$)"
        synthesis_match = re.search(synthesis_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if synthesis_match:
            steps['synthesis'] = synthesis_match.group(1).strip()
        
        return steps
    
    def extract_supporting_facts(self, response: str) -> List[tuple]:
        """
        Extract supporting facts (document, sentence) from response
        """
        supporting_facts = []
        
        # Look for document and sentence references
        pattern = r"Document:\s*(.+?)\s*Sentence Reference:\s*\[?(\d+)\]?"
        matches = re.finditer(pattern, response, re.IGNORECASE)
        
        for match in matches:
            doc_title = match.group(1).strip()
            sent_id = int(match.group(2))
            supporting_facts.append((doc_title, sent_id))
        
        return supporting_facts
    
    def majority_vote(self, responses: List[str]) -> str:
        """
        Perform majority voting on multiple responses
        """
        answers = []
        
        for response in responses:
            answer = self.extract_answer(response)
            if answer:
                # Normalize answer for comparison
                normalized = answer.lower().strip()
                answers.append(normalized)
        
        if not answers:
            return ""
        
        # Count occurrences
        counter = Counter(answers)
        most_common = counter.most_common(1)[0][0]
        
        return most_common
    
    def calculate_confidence(
        self,
        response: str,
        expected_hops: int = 2
    ) -> float:
        """
        Estimate confidence in the answer
        """
        score = 0.0
        
        # Check if answer is present
        if self.extract_answer(response):
            score += 0.3
        
        # Check if reasoning steps are present
        steps = self.extract_reasoning_steps(response)
        if len(steps) >= expected_hops:
            score += 0.3
        
        # Check if supporting facts are cited
        facts = self.extract_supporting_facts(response)
        if len(facts) >= expected_hops:
            score += 0.2
        
        # Check response length (not too short, not too long)
        word_count = len(response.split())
        if 100 <= word_count <= 1000:
            score += 0.2
        
        return min(score, 1.0)