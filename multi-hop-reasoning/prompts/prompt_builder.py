"""Build prompts for multi-hop reasoning"""
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class PromptComponents:
    system_instructions: str
    few_shot_examples: str
    reasoning_instructions: str
    test_question: str
    format_specification: str

class PromptBuilder:
    def __init__(self):
        self.bridge_template = self._get_bridge_template()
        self.comparison_template = self._get_comparison_template()
    
    def build_prompt(
        self,
        question: str,
        context: Dict[str, List[str]],
        question_type: str,
        level: str,
        similar_examples: List,
        supporting_facts_hint: Optional[List[str]] = None
    ) -> str:
        components = PromptComponents(
            system_instructions=self._build_system_instructions(question_type),
            few_shot_examples=self._build_few_shot_examples(similar_examples),
            reasoning_instructions=self._build_reasoning_instructions(question_type),
            test_question=self._build_test_question(question, context, supporting_facts_hint),
            format_specification=self._build_format_specification()
        )
        
        prompt = f"""{components.system_instructions}

{components.few_shot_examples}

{components.reasoning_instructions}

{components.test_question}

{components.format_specification}"""
        
        return prompt
    
    def _build_system_instructions(self, question_type: str) -> str:
        template = self.bridge_template if question_type == 'bridge' else self.comparison_template
        
        instructions = f"""You are an expert at multi-hop reasoning. Your task is to answer questions that require combining information from multiple documents.

IMPORTANT PRINCIPLES:
1. All information needed to answer the question is in the provided context
2. You must trace your reasoning through specific sentences
3. Each claim must be supported by citing the exact sentence
4. Follow the reasoning pattern step-by-step

REASONING PATTERN FOR {question_type.upper()} QUESTIONS:
{template}"""
        
        return instructions
    
    def _build_few_shot_examples(self, similar_examples: List) -> str:
        if not similar_examples:
            return ""
        
        examples_text = "HERE ARE EXAMPLES OF CORRECT REASONING:\n\n"
        
        for idx, example in enumerate(similar_examples[:3], 1):
            pattern = example.pattern
            
            examples_text += f"""{'='*80}
EXAMPLE {idx}:
{'='*80}

QUESTION: {pattern.question}

AVAILABLE DOCUMENTS:
{self._format_document_titles(pattern)}

CONTEXT:
{self._format_context_for_example(pattern)}

REASONING PROCESS:

"""
            
            for hop_idx, hop in enumerate(pattern.hops, 1):
                examples_text += f"""HOP {hop_idx}:
Document: {hop.document_title}
Sentence [{hop.sentence_id}]: "{hop.sentence_text}"
Role: {hop.role}
Key Information: {self._extract_key_info(hop)}

"""
            
            examples_text += f"""REASONING CHAIN:
{self._build_reasoning_chain(pattern)}

FINAL ANSWER: {pattern.answer}

"""
        
        return examples_text
    
    def _build_reasoning_instructions(self, question_type: str) -> str:
        if question_type == 'bridge':
            instructions = """
NOW, FOLLOW THESE STEPS FOR THE TEST QUESTION:

Step 1 - QUESTION ANALYSIS:
- What is the question asking for?
- What type of information do you need?
- Are there any key entities or concepts mentioned?

Step 2 - DOCUMENT SELECTION:
- Look at the document titles
- Which documents are likely to contain relevant information?

Step 3 - HOP 1 (Find the Bridge):
- Identify the primary entity in the question
- Find the document about this entity
- Extract the key relationship or property
- Identify any secondary entity or concept mentioned

Step 4 - HOP 2 (Follow the Bridge):
- Look for documents about the secondary entity
- Find the specific information requested in the question
- Verify this answers what was asked

Step 5 - SYNTHESIZE ANSWER:
- Combine information from both hops
- Formulate a clear, direct answer
"""
        else:
            instructions = """
NOW, FOLLOW THESE STEPS FOR THE TEST QUESTION:

Step 1 - QUESTION ANALYSIS:
- What two entities are being compared?
- What property or dimension is being compared (date, size, quantity, etc.)?
- What comparison operator is used (first, larger, more, etc.)?

Step 2 - DOCUMENT SELECTION:
- Find documents about Entity A
- Find documents about Entity B

Step 3 - HOP 1 (Entity A):
- Locate the document about Entity A
- Find the sentence containing the comparison property
- Extract the value or property

Step 4 - HOP 2 (Entity B):
- Locate the document about Entity B
- Find the sentence containing the comparison property
- Extract the value or property

Step 5 - COMPARE AND ANSWER:
- Compare the two values/properties
- Apply the comparison operator
- State which entity satisfies the comparison
"""
        
        return instructions
    
    def _build_test_question(
        self,
        question: str,
        context: Dict[str, List[str]],
        supporting_facts_hint: Optional[List[str]] = None
    ) -> str:
        test_section = f"""{'='*80}
YOUR TASK:
{'='*80}

QUESTION: {question}

"""
        
        if supporting_facts_hint and len(supporting_facts_hint) > 0:
            test_section += f"""HINT: The answer requires information from these documents:
{chr(10).join(f'- {doc}' for doc in supporting_facts_hint)}

"""
        
        test_section += f"""AVAILABLE DOCUMENTS:
{chr(10).join(f'{idx+1}. {title}' for idx, title in enumerate(context.keys()))}

FULL CONTEXT:

"""
        
        for doc_idx, (title, sentences) in enumerate(context.items(), 1):
            test_section += f"""Document {doc_idx}: {title}
"""
            for sent_idx, sentence in enumerate(sentences):
                test_section += f"[{sent_idx}] {sentence}\n"
            test_section += "\n"
        
        return test_section
    
    def _build_format_specification(self) -> str:
        format_spec = """
YOU MUST RESPOND IN THIS EXACT FORMAT:

REASONING PROCESS:

Step 1 - Question Analysis:
[What is being asked? What information is needed?]

Step 2 - Document Selection:
[Which documents are relevant?]

Step 3 - HOP 1:
Document: [document title]
Sentence Reference: [sentence index]
Sentence Text: "[exact sentence from context]"
Information Extracted: [key fact]
Relevance: [why this helps]

Step 4 - HOP 2:
Document: [document title]
Sentence Reference: [sentence index]
Sentence Text: "[exact sentence from context]"
Information Extracted: [key fact]
Connection to HOP 1: [how this connects]

[Add more hops if needed]

Step N - Answer Synthesis:
[Combine the information to form answer]

FINAL ANSWER: [Give ONLY the direct answer - a name, date, place, or short phrase. Do NOT include explanations, full sentences, or phrases like "based on" or "combining the information".]

EXAMPLES OF GOOD FINAL ANSWERS:
- "Javier Sotomayor"
- "Alex Cox"
- "Hurricane Ivan"
- "four times"
- "Newton Heath LYR Football Club"

EXAMPLES OF BAD FINAL ANSWERS (DO NOT DO THIS):
- "Based on the information, the answer is..."
- "Combining the information from both hops..."
- "We can conclude that..."
- "The answer is clearly..."
"""
        
        return format_spec
    
    def _get_bridge_template(self) -> str:
        return """
Bridge questions require finding an intermediate entity:
1. Question mentions Entity A and asks about property P
2. Find document about Entity A → Extract relationship to Entity B
3. Find document about Entity B → Extract property P
4. Entity B is the "bridge" connecting A to the answer
"""
    
    def _get_comparison_template(self) -> str:
        return """
Comparison questions require finding properties of two entities:
1. Question asks to compare Entity A and Entity B on dimension D
2. Find document about Entity A → Extract dimension D value
3. Find document about Entity B → Extract dimension D value
4. Compare values and determine which satisfies the comparison operator
"""
    
    def _format_document_titles(self, pattern) -> str:
        titles = list(set(hop.document_title for hop in pattern.hops))
        return "\n".join(f"- {title}" for title in titles)
    
    def _format_context_for_example(self, pattern) -> str:
        docs = {}
        for hop in pattern.hops:
            if hop.document_title not in docs:
                docs[hop.document_title] = []
            docs[hop.document_title].append((hop.sentence_id, hop.sentence_text))
        
        result = ""
        for doc_title, sentences in docs.items():
            result += f"\nDocument: {doc_title}\n"
            for sent_id, sent_text in sorted(sentences):
                result += f"[{sent_id}] {sent_text}\n"
        
        return result
    
    def _extract_key_info(self, hop) -> str:
        if hop.entities_extracted:
            return f"Mentions: {', '.join(hop.entities_extracted[:3])}"
        return "Core fact for reasoning chain"
    
    def _build_reasoning_chain(self, pattern) -> str:
        if len(pattern.hops) == 0:
            return "No reasoning chain available"
        
        chain = []
        for idx, hop in enumerate(pattern.hops, 1):
            chain.append(f"Hop {idx} tells us: [from {hop.document_title}] {hop.sentence_text[:100]}...")
        
        if pattern.bridge_entity:
            chain.append(f"The bridge entity '{pattern.bridge_entity}' connects the hops")
        
        chain.append(f"Therefore, the answer is: {pattern.answer}")
        
        return "\n".join(chain)