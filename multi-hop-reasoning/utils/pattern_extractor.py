"""Extract reasoning patterns from training examples"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re

@dataclass
class HopInfo:
    document_title: str
    sentence_id: int
    sentence_text: str
    entities_extracted: List[str]
    role: str

@dataclass
class ReasoningPattern:
    question_id: str
    question: str
    question_type: str
    level: str
    num_hops: int
    hops: List[HopInfo]
    bridge_entity: Optional[str] = None
    comparison_dimension: Optional[str] = None
    answer: str = ""

class PatternExtractor:
    def extract_pattern(self, question_id: str, question: str, question_type: str, 
                       level: str, supporting_facts: List[Tuple[str, int]], 
                       context: Dict[str, List[str]], answer: str) -> ReasoningPattern:
        hops = []
        
        for idx, (doc_title, sent_id) in enumerate(supporting_facts):
            if doc_title in context:
                sentences = context[doc_title]
                if sent_id < len(sentences):
                    sentence_text = sentences[sent_id]
                    entities = self._extract_entities(sentence_text)
                    role = self._determine_hop_role(idx, len(supporting_facts), question_type)
                    
                    hop = HopInfo(
                        document_title=doc_title,
                        sentence_id=sent_id,
                        sentence_text=sentence_text,
                        entities_extracted=entities,
                        role=role
                    )
                    hops.append(hop)
        
        bridge_entity = None
        if question_type == 'bridge' and len(hops) >= 2:
            bridge_entity = self._find_bridge_entity(hops)
        
        comparison_dim = None
        if question_type == 'comparison':
            comparison_dim = self._find_comparison_dimension(question)
        
        return ReasoningPattern(
            question_id=question_id,
            question=question,
            question_type=question_type,
            level=level,
            num_hops=len(hops),
            hops=hops,
            bridge_entity=bridge_entity,
            comparison_dimension=comparison_dim,
            answer=answer
        )
    
    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        quoted = re.findall(r'"([^"]*)"', text)
        entities.extend(quoted)
        
        words = text.split()
        current_entity = []
        for word in words:
            if word and word[0].isupper():
                current_entity.append(word)
            else:
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
        
        if current_entity:
            entities.append(' '.join(current_entity))
        
        return list(set(entities))
    
    def _determine_hop_role(self, hop_idx: int, total_hops: int, question_type: str) -> str:
        if question_type == 'bridge':
            if hop_idx == 0:
                return 'starter'
            elif hop_idx == total_hops - 1:
                return 'final'
            else:
                return 'bridge'
        else:
            return 'comparison'
    
    def _find_bridge_entity(self, hops: List[HopInfo]) -> Optional[str]:
        if len(hops) < 2:
            return None
        
        entities_1 = set(hops[0].entities_extracted)
        entities_2 = set(hops[1].entities_extracted)
        common = entities_1.intersection(entities_2)
        
        if common:
            return list(common)[0]
        
        for entity in entities_1:
            if entity.lower() in hops[1].sentence_text.lower():
                return entity
        
        return None
    
    def _find_comparison_dimension(self, question: str) -> Optional[str]:
        question_lower = question.lower()
        dimensions = {
            'date': ['first', 'earlier', 'before', 'founded', 'started', 'established'],
            'size': ['larger', 'bigger', 'smaller', 'size', 'area'],
            'quantity': ['more', 'less', 'most', 'least', 'number'],
            'location': ['where', 'located', 'place'],
            'duration': ['longer', 'shorter', 'duration', 'time'],
        }
        
        for dim, keywords in dimensions.items():
            if any(kw in question_lower for kw in keywords):
                return dim
        
        return 'unknown'