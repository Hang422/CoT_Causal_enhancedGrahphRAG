from typing import Dict, List, Optional
import logging
import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict
from src.modules.MedicalQuestion import MedicalQuestion
from config import config


@dataclass
class LLMInteraction:
    """Data class to store LLM interaction details"""
    timestamp: str
    model: str
    interaction_type: str
    question_id: str
    question_text: str
    messages: List[Dict]
    response: str
    metadata: Dict

    def to_dict(self) -> Dict:
        """Convert interaction to dictionary format"""
        return asdict(self)


def _generate_question_identifier(question: str) -> str:
    """Generate a unique identifier for the question"""
    return hashlib.md5(question.encode()).hexdigest()[:8]


class LLMLogger:
    """Enhanced logging system for LLM interactions"""

    def __init__(self):
        """Initialize logger with configured paths and interaction types"""
        self.base_log_dir = config.paths["logs"] / "llm_logs"
        self.base_log_dir.mkdir(parents=True, exist_ok=True)

        # Model-specific logging directories
        self.models = {
            'gpt-3.5-turbo': 'gpt35',
            'gpt-3.5-turbo-16k': 'gpt35',
            'gpt-4-turbo': 'gpt4',
            'gpt-4': 'gpt4',
            'gpt-4o': 'gpt4o',
            'gpt-4o-mini': 'gpt4o-mini'
        }

        # Interaction types
        self.interaction_types = {
            'direct_answer': 'direct_answer',
            'reasoning_chain': 'reasoning_chain',
            'enhancement': 'enhancement',
            'final_answer': 'final_answer',
            'chain_of_thought': 'chain_of_thought'
        }

        self._create_log_directories()
        self.interaction_counts = {itype: 0 for itype in self.interaction_types}

    def _create_log_directories(self) -> None:
        """Create hierarchical directory structure for logs"""
        self.log_dirs = {}
        timestamp = datetime.now().strftime("%Y%m%d")

        model_types = {'gpt35', 'gpt4', 'gpt4o'}

        for model in model_types:
            model_dir = self.base_log_dir / timestamp / model
            for interaction in self.interaction_types:
                dir_path = model_dir / interaction
                dir_path.mkdir(parents=True, exist_ok=True)
                self.log_dirs[(model, interaction)] = dir_path

    def _get_model_name(self, model: str) -> str:
        """Get standardized model name for logging"""
        return self.models.get(model, 'gpt35')

    def log_interaction(self, model: str, interaction_type: str,
                        question: MedicalQuestion, messages: List[Dict],
                        response: str, metadata: Optional[Dict] = None) -> None:
        """Log an LLM interaction with enhanced metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self._get_model_name(model)
            question_id = _generate_question_identifier(question.question)

            interaction = LLMInteraction(
                timestamp=timestamp,
                model=model_name,
                interaction_type=interaction_type,
                question_id=question_id,
                question_text=question.question,
                messages=messages,
                response=response,
                metadata={
                    'topic': question.topic_name,
                    'is_multi_choice': question.is_multi_choice,
                    'has_reasoning_chain': bool(question.reasoning_chain),
                    'has_enhanced_info': bool(question.enhanced_information),
                    'model_config': config.openai.get("model"),
                    'temperature': config.openai.get("temperature"),
                    **(metadata or {})
                }
            )

            log_dir = self.log_dirs.get((model_name, interaction_type))
            if not log_dir:
                raise ValueError(f"Invalid model ({model}) or interaction type ({interaction_type})")

            log_file = log_dir / f"{timestamp}_{question_id}.json"

            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(interaction.to_dict(), f, ensure_ascii=False, indent=2)

            self.interaction_counts[interaction_type] += 1

        except Exception as e:
            logger = config.get_logger("llm_logger")
            logger.error(f"Error logging interaction: {str(e)}", exc_info=True)

            error_dir = self.base_log_dir / "errors"
            error_dir.mkdir(exist_ok=True)

            error_file = error_dir / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                        'error': str(e),
                        'model': model,
                        'interaction_type': interaction_type,
                        'question_id': question_id if 'question_id' in locals() else None
                    }, f, ensure_ascii=False, indent=2)
            except Exception:
                pass


def clean_json_response(response: str) -> Dict:
    """Clean and parse JSON response from LLM"""
    if not response or not isinstance(response, str):
        raise ValueError("Input must be a non-empty string")

    cleaned = response.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}') + 1
    if start_idx == -1 or end_idx <= start_idx:
        raise ValueError("No valid JSON object found")

    json_str = cleaned[start_idx:end_idx]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str = ' '.join(json_str.replace('\n', ' ').replace('\r', ' ').split())
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON after cleaning: {str(e)}")


def _update_question_with_error(question: MedicalQuestion) -> None:
    """Update question with error state"""
    question.analysis = "Error processing answer"
    question.answer = ""
    question.confidence = 0.0


def _update_question_from_result(question: MedicalQuestion, result: Dict) -> None:
    """Update question with standardized result processing"""
    question.analysis = result.get("final_analysis", "")
    question.answer = result.get("answer", "").lower()
    try:
        question.confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        question.confidence = 0.0


class LLMProcessor:
    """Handles all LLM interactions with standardized processing"""

    def __init__(self):
        self.client = OpenAI(api_key=config.openai["api_key"])
        self.model = config.openai.get("model")
        self.model_name = self._get_model_name()
        self.temperature = config.openai.get("temperature")
        self.logger = LLMLogger()
        config.get_logger("question_processor").info(f"Initialized model {self.model}")

    def _get_model_name(self) -> str:
        """Get shortened model name for logging"""
        model_mapping = {
            'gpt-3.5-turbo': 'gpt35',
            'gpt-3.5-turbo-16k': 'gpt35',
            'gpt-4-turbo': 'gpt4',
            'gpt-4': 'gpt4',
            'gpt-4o': 'gpt4o',
            'gpt-4o-mini': 'gpt4o-mini'
        }
        return model_mapping.get(self.model, 'gpt4')

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _get_completion(self, messages: List[Dict], interaction_type: str,
                        question: MedicalQuestion) -> Dict:
        """Get standardized completion from LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            response_text = response.choices[0].message.content

            # Log the interaction
            self.logger.log_interaction(
                model=self.model_name,
                interaction_type=interaction_type,
                question=question,
                messages=messages,
                response=response_text
            )

            # Clean and validate response
            return clean_json_response(response_text)

        except Exception as e:
            logging.error(f"Error in LLM call: {str(e)}")
            raise

    def direct_answer(self, question: MedicalQuestion) -> None:
        """Direct answer implementation with standardized processing"""
        if question.topic_name is not None:
            prompt = f"""You are a medical expert specializing in the field of {question.topic_name}."""
        else:
            prompt = """You are a medical expert."""

        prompt += f"""Please help answer this {'' if not question.is_multi_choice else 'multiple choice'} question:

Question: {question.question}
"""
        if question.is_multi_choice:
            prompt += "\nOptions:\n"
            for key, text in question.options.items():
                prompt += f"{key}. {text}\n"

        prompt += """
        ### Output Format
        Provide your response in valid JSON format:
        {   
            "final_analysis": "Your concise analysis following the above structure",
            "answer": "Option key from the available options (only key,like opa)",
            "confidence": Score between 0-100 based on alignment with established medical facts
        }
        """

        try:
            messages = [
                {"role": "system", "content": "You are a medical expert making decisions based on medical knowledge."},
                {"role": "user", "content": prompt}
            ]

            result = self._get_completion(messages, "direct_answer", question)
            _update_question_from_result(question, result)

        except Exception as e:
            logging.error(f"Error in direct answer: {str(e)}")
            _update_question_with_error(question)

    def generate_reasoning_chain(self, question: MedicalQuestion) -> None:
        """Generate reasoning chains for medical diagnosis"""
        prompt = f"""As an expert reasoner, analyze this question by generating multiple reasoning chains. Each chain must follow strict structural rules to ensure clarity and verifiability.

    ### Question
    {question.question}

    ### Options
    """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        prompt += """
    ### Key Requirements for Reasoning Chains:

    1. **Node Structure**:
       - Each node must be a single fact, entity, or state.
       - No reasoning or relationships should exist within nodes.
       - Example of Incorrect Node: "Insulin lowers blood glucose."
       - Example of Correct Chain: "Insulin" -> "Blood glucose decreases."

    2. **Chain Structure**:
       - Connect nodes with "->" to show logical progression.
       - Each arrow represents a causal or logical relationship.
       - Include intermediary steps to show complete logical paths.
       - Example: "Drug A" -> "Receptor binding" -> "Signal cascade activation" -> "Clinical effect."

    3. **Uncertainty Handling**:
       - Explicitly indicate when verification or additional evidence is required.
       - If a connection is uncertain, add "Requires verification" or specify missing evidence.
       - Example: "Compound X" -> "Potential receptor binding" -> "Requires verification -> Conclusion unclear."

    4. **Output Format**:
    Provide reasoning chains in the following JSON format:
    {
        "reasoning_chains": [
            "Entity A -> Intermediate step -> Missing evidence for [specific aspect] -> Conclusion for option X",
            "Different entity -> Different pathway -> Requires verification -> Conclusion for option Y"
        ]
    }

    ### Guidelines:
    - Generate **multiple independent reasoning chains**.
    - Each chain should focus on one specific aspect or pathway.
    - Include both supporting and opposing chains for all options.
    - Clearly identify uncertainties without making unjustified assumptions.
    - Avoid redundancy by keeping nodes factual and distinct.
    - Let the connections (arrows) explain the reasoning, not the nodes themselves.

    Remember: The reasoning is represented in the arrows (->) between nodes. Nodes themselves are facts, not explanations.
    """

        try:
            messages = [
                {"role": "system", "content": "You are a medical expert generating diagnostic reasoning chains."},
                {"role": "user", "content": prompt}
            ]

            result = self._get_completion(messages, "reasoning_chain", question)
            question.reasoning_chain = result.get("reasoning_chains", [])

        except Exception as e:
            logging.error(f"Error in reasoning chain generation: {str(e)}")
            question.reasoning_chain = []

    def enhance_information(self, question: MedicalQuestion) -> None:
        """Enhancement with standardized processing"""
        if question.topic_name is not None:
            prompt = f"You are a medical expert specializing in the field of {question.topic_name}."
        else:
            prompt = f"You are a medical expert."

        prompt += f"""
        Your task: Integrate, refine, and validate the provided information (reasoning chains, causal graphs, and knowledge graphs) into a clinically accurate and directly relevant enhanced summary. Critically evaluate the reasoning chains for correctness and identify any areas needing further validation. Ensure the final output is concise, medically accurate, and supports reasoning towards the question without providing the final answer.

        ### Inputs
        **Question**: {question.question}

        **Options**:
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        if question.reasoning_chain:
            prompt += "\n### Reasoning Chains (Model-Generated)\n"
            for chain in question.reasoning_chain:
                prompt += f"- {chain}\n"

        if len(question.enhanced_graph.paths) > 0:
            prompt += "\n### Auxiliary Information (Retrieved Graph Data):\n"
            for path in question.enhanced_graph.paths:
                prompt += f"- {path}\n"

        prompt += """
        ### Task Requirements

        1. **Evaluate Reasoning Chains**:
           - Identify and mark chains as correct, partially correct, or incorrect based on standard medical knowledge.
           - Provide a concise explanation for why a chain is correct or incorrect, or what further validation is needed.

        2. **Filter and Curate Information**:
           - Remove irrelevant, overly generic, or misleading relationships from the graph data.
           - Retain only information that is:
             - Medically accurate (aligned with standard clinical knowledge).
             - Directly relevant to differentiating the options or clarifying the question.
             - Useful in improving reasoning without solving the question outright.

        3. **Fuse and Enhance Information**:
           - Combine relevant reasoning chains and graph data into a cohesive, concise summary.
           - Ensure the summary is directly applicable to supporting reasoning for the question.

        4. **Quality Control**:
           - Verify that all retained information is accurate, relevant, and contributes to clinical understanding.
           - Avoid providing or hinting at the final answer.

        ### Output Format
        Provide your response in the following JSON format:
        {
          "enhanced_information": "Your carefully curated and clinically relevant summary here"
        }
        """

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert making decisions based on enhanced information."},
                {"role": "user", "content": prompt}
            ]

            result = self._get_completion(messages, "enhancement", question)
            question.enhanced_information = result.get("enhanced_information", "")

        except Exception as e:
            logging.error(f"Error in enhancement: {str(e)}")
            question.enhanced_information = ""

    def answer_with_enhanced_information(self, question: MedicalQuestion) -> None:
        """Final answer with standardized processing"""
        if question.topic_name is not None:
            prompt = f"You are a medical expert specializing in the field of {question.topic_name}."
        else:
            prompt = f"You are a medical expert."

        if len(question.enhanced_graph.paths) > 0:
            prompt += f"""Please help answer this {'' if not question.is_multi_choice else 'multiple choice'} question:

            Question: {question.question}
            """
            if question.is_multi_choice:
                prompt += "\nOptions:\n"
                for key, text in question.options.items():
                    prompt += f"{key}. {text}\n"

            prompt += """
                    ### Output Format
                    Provide your response in valid JSON format:
                    {   
                        "final_analysis": "Your concise analysis following the above structure",
                        "answer": "Option key from the available options (only key,like opa)",
                        "confidence": Score between 0-100 based on alignment with established medical facts
                    }
                    """
        else:
            prompt += f"""
            Your task: Accurately answer the question based on a combination of standard medical knowledge (primary source) and enhanced information (secondary source). Ensure the final answer is clinically valid, logically derived, and explicitly justified.
    
            ### Question
            {question.question}
    
            ### Options
            """
            for key, text in question.options.items():
                prompt += f"{key}. {text}\n"

            if question.enhanced_information:
                prompt += f"\n### Enhanced Information (For Contextual Support):\n{question.enhanced_information}\n"

            prompt += """
            ### Task Requirements
    
            1. **Analysis Framework**:
               - Identify the core medical issue or concept within the question.
               - Use standard medical knowledge to verify the most likely answer based on established facts.
               - Incorporate relevant enhanced information only if it aligns with medical consensus.
               - If the enhanced information contains contradictions or irrelevant details, prioritize standard medical facts and ignore the conflicting parts.
    
            2. **Decision Framework**:
               - Always provide a definitive answer (there is one correct answer).
               - Base the answer on the simplest and most direct medical reasoning.
               - Use enhanced information to strengthen confidence in the answer, but not as the sole determinant.
    
            3. **Reasoning Expectations**:
               - Explain the reasoning for the chosen answer step by step.
               - Highlight the key medical facts or enhanced information that led to your conclusion.
               - Discard irrelevant or misleading details from the enhanced information.
    
            4. **Output Format and Confidence**:
               - Confidence should reflect how strongly the chosen answer aligns with standard medical knowledge and how well it is supported by the enhanced information.
               - Lower confidence indicates partial reliance on enhanced information or remaining ambiguities.
    
            ### Output Format
            Provide your response in valid JSON format:
            {
                "final_analysis": "Your clear and concise analysis, step-by-step reasoning based on standard medical knowledge and enhanced information.",
                "answer": "Option key from the available options (e.g., opa, opb, etc.)",
                "confidence": A score between 0-100 indicating the strength of your answer
            }
    
            ### Key Guidelines
            - **Be Clinically Accurate**: Align your answer with what a practicing clinician would consider correct.
            - **Focus on Relevance**: Use enhanced information only when it supports or clarifies the medical consensus.
            - **Avoid Overcomplication**: Present clear, step-by-step reasoning without unnecessary complexity.
            """

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert making decisions based on enhanced information."},
                {"role": "user", "content": prompt}
            ]

            result = self._get_completion(messages, "final_answer", question)
            _update_question_from_result(question, result)

        except Exception as e:
            logging.error(f"Error in final answer: {str(e)}")
            _update_question_with_error(question)

    def answer_with_cot(self, question: MedicalQuestion) -> None:
        """Answer using chain-of-thought reasoning with graph paths and reasoning chains"""
        if question.topic_name is not None:
            prompt = f"You are a medical expert specializing in the field of {question.topic_name}."
        else:
            prompt = f"You are a medical expert."

        prompt += f"""
        Your task: Analyze and answer the question using chain-of-thought reasoning based on the provided reasoning chains and graph paths. Do not use any enhanced information.

        ### Question
        {question.question}

        ### Options
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        if question.reasoning_chain:
            prompt += "\n### Reasoning Chains:\n"
            for chain in question.reasoning_chain:
                prompt += f"- {chain}\n"

        if question.enhanced_graph and question.enhanced_graph.paths:
            prompt += "\n### Graph Paths:\n"
            for path in question.enhanced_graph.paths:
                prompt += f"- {path}\n"

        prompt += """
        ### Task Requirements

        1. **Chain-of-Thought Analysis**:
           - Analyze the reasoning chains and graph paths systematically.
           - Identify which chains and paths are most relevant to answering the question.
           - Explain how different pieces of evidence connect and support each other.
           - Note any contradictions or gaps in the reasoning.

        2. **Evidence Integration**:
           - Combine evidence from reasoning chains and graph paths.
           - Explain how different pieces of evidence support or contradict each other.
           - Identify the strongest line of reasoning that leads to your answer.

        3. **Medical Knowledge Application**:
           - Use standard medical knowledge to validate the reasoning chains and paths.
           - Explain how the evidence aligns with established medical principles.
           - Note when the evidence contradicts known medical facts.

        4. **Decision Making**:
           - Explain your step-by-step thought process.
           - Show how you weigh different pieces of evidence.
           - Justify your final choice based on the strongest evidence.

        ### Output Format
        Provide your response in valid JSON format:
        {
            "chain_of_thought": "Your detailed step-by-step reasoning process, showing how you integrated the reasoning chains and graph paths",
            "final_analysis": "Your concise conclusion based on the chain of thought",
            "answer": "Option key (e.g., opa)",
            "confidence": Score between 0-100 based on the strength of the reasoning chain and supporting evidence
        }

        ### Key Guidelines
        - **Be Systematic**: Show clear progression in your reasoning.
        - **Be Explicit**: Explain how each piece of evidence contributes to your conclusion.
        - **Be Critical**: Acknowledge limitations and uncertainties in the reasoning.
        """

        try:
            messages = [
                {"role": "system", "content": "You are a medical expert using chain-of-thought reasoning."},
                {"role": "user", "content": prompt}
            ]

            result = self._get_completion(messages, "chain_of_thought", question)

            # Update question with COT-specific information
            question.chain_of_thought = result.get("chain_of_thought", "")
            question.analysis = result.get("final_analysis", "")
            question.answer = result.get("answer", "").lower()
            try:
                question.confidence = float(result.get("confidence", 0.0))
            except (TypeError, ValueError):
                question.confidence = 0.0

        except Exception as e:
            logging.error(f"Error in chain-of-thought answer: {str(e)}")
            question.chain_of_thought = ""
            _update_question_with_error(question)
