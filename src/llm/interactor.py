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
            'gpt-3.5-turbo': 'gpt-3.5-turbo',
            'gpt-4-turbo': 'gpt-4-turbo',
            'gpt-4o': 'gpt-4o',
            'gpt-4o-mini': 'gpt-4o-mini'
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

        # Explicit list of model types
        model_types = {'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini'}

        for model in model_types:
            model_dir = self.base_log_dir / timestamp / model
            for interaction in self.interaction_types:
                dir_path = model_dir / interaction
                dir_path.mkdir(parents=True, exist_ok=True)
                self.log_dirs[(model, interaction)] = dir_path

    def _get_model_name(self, model: str) -> str:
        """Get standardized model name for logging"""
        return self.models.get(model, 'gpt-3.5-turbo')  # Default to gpt-3.5-turbo if unknown model

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


def _get_model_name(model: str) -> str:
    """Return the standardized model name."""
    model_mapping = {
        'gpt-3.5-turbo': 'gpt-3.5-turbo',
        'gpt-4-turbo': 'gpt-4-turbo',
        'gpt-4o': 'gpt-4o',
        'gpt-4o-mini': 'gpt-4o-mini'
    }
    return model_mapping.get(model, 'gpt-3.5-turbo')  # Default to gpt-3.5-turbo if unknown model


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
        self.model_name = config.openai.get("model")
        self.temperature = config.openai.get("temperature")
        self.logger = LLMLogger()
        config.get_logger("question_processor").info(f"Initialized model {self.model}")

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
        """Generate reasoning chains for medical diagnosis with enhanced response parsing"""
        prompt = f"""As a medical expert, analyze this question by generating multiple reasoning chains. Each chain consists of steps connected by reasoning arrows, where each step can show the result of previous reasoning but cannot contain complete entity-to-entity reasoning within itself.

        ### Question
        {question.question}

        ### Options
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        prompt += """
        ### Key Requirements for Reasoning Chains:

        1. **Step Content Rules**:
           - A step can contain reasoning results from previous step
           - A step must not contain a complete entity-to-entity reasoning within itself
           - Example correct: "Insulin" -> "decreased blood glucose"
           - Example correct: "decreased blood glucose" -> "improved metabolic state"
           - Example incorrect: "Insulin decreases blood glucose" (complete reasoning in one step)
           - Example incorrect: "decreased blood glucose which then improves metabolism" (contains next reasoning step)

        2. **Chain Structure**:
           - Arrows (->) represent the reasoning process between steps
           - Each new step builds on previous reasoning result
           - Example chain: 
             "Insulin" -> "decreased blood glucose" -> "stable metabolic state" -> 90%
           - Complex example:
             "ACE inhibitors" -> "reduced Angiotensin II" -> "lowered blood pressure" -> 85%

        3. **Uncertainty and Confidence**:
           - Mark uncertain reasonings explicitly
           - Example with uncertainty:
             "Drug X" -> "possible receptor blockade" -> "target response unclear" -> 40%
           - End each chain with overall confidence (0-100%)
           - Lower confidence for chains with uncertain steps

        ### Output Format
        Please provide each reasoning chain on a new line starting with CHAIN:

        Example format:
        CHAIN: Entity1 -> ReasoningResult1 -> FurtherResult -> 90%
        CHAIN: EntityA -> UncertainResult -> PossibleOutcome -> 60%

        ### Key Guidelines:
        - Steps can show results of previous reasoning
        - No complete entity-to-entity reasoning within a single step
        - The arrow (->) shows how we reason from one step to the next
        - Keep each step focused on one result or state
        - Include uncertainty where it exists
        - End chains with confidence percentage

        Remember:
        - A step can contain the result of reasoning (e.g., "decreased blood glucose")
        - But it cannot contain a complete reasoning process (e.g., "insulin reduces blood glucose")
        - Example: "Insulin" -> "decreased blood glucose" -> "metabolic improvement" -> 90%
          * First arrow represents how Insulin affects blood glucose
          * Second arrow represents how the decreased blood glucose leads to metabolic improvement
          * Each step shows the result of previous reasoning without including the next reasoning step
        """

        try:
            messages = [
                {"role": "system", "content": "You are a medical expert generating diagnostic reasoning chains."},
                {"role": "user", "content": prompt}
            ]

            # Get the raw completion
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            response_text = response.choices[0].message.content

            # Log the interaction before parsing
            self.logger.log_interaction(
                model=self.model_name,
                interaction_type="reasoning_chain",
                question=question,
                messages=messages,
                response=response_text
            )

            # Parse the response text to extract chains
            chains = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('CHAIN:'):
                    # Extract the chain part after 'CHAIN:'
                    chain = line[6:].strip()
                    # Basic validation: check if chain has the required format
                    if ('->' in chain and
                            chain.count('->') >= 2 and  # At least two arrows
                            '%' in chain.split('->')[-1]):  # Ends with confidence
                        chains.append(chain)

            # If no valid chains were found, try alternative parsing
            if not chains:
                # Look for chains that might not have the CHAIN: prefix
                lines = [line.strip() for line in response_text.split('\n') if '->' in line]
                for line in lines:
                    if (line.count('->') >= 2 and  # At least two arrows
                            '%' in line.split('->')[-1]):  # Ends with confidence
                        chains.append(line)

            # Update the question with found chains, or empty list if none found
            question.reasoning_chain = chains if chains else []

        except Exception as e:
            logging.error(f"Error in reasoning chain generation: {str(e)}", exc_info=True)
            question.reasoning_chain = []

    def enhance_information(self, question: MedicalQuestion) -> None:
        """Enhancement with standardized processing"""
        if len(question.enhanced_graph.paths) == 0:
            return
        if question.topic_name is not None:
            prompt = f"You are a medical expert specializing in the field of {question.topic_name}."
        else:
            prompt = f"You are a medical expert."

        prompt += f"""
        Your task: Provide a concise and clinically relevant summary that integrates multiple sources of evidence: reasoning chains, retrieved path evidence, and standard medical knowledge. Focus on decision-critical findings, prioritize the strongest reasoning chains, and address contradictions or gaps in evidence.

        ### Question
        {question.question}

        ### Options
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        if question.reasoning_chain:
            prompt += "\n### Reasoning Chains to Evaluate:\n"
            for chain in question.reasoning_chain:
                prompt += f"- {chain}\n"

        if question.enhanced_graph and len(question.enhanced_graph.paths) > 0:
            prompt += "\n### Supporting Evidence (Retrieved Paths):\n"
            for path in question.enhanced_graph.paths:
                prompt += f"- {path}\n"

        prompt += """
        ### Key Principles for Evaluation:
        1. Use retrieved paths and reasoning chains to validate or refute possible diagnoses.
        2. Absence of path evidence does not invalidate a reasoning step, but it may reduce confidence.
        3. Use medical knowledge to:
           - Strengthen reasoning chains supported by evidence.
           - Evaluate reasoning steps without direct evidence.
           - Highlight contradictions or weak points in reasoning.
        4. Prioritize diagnoses with the strongest combined support from reasoning chains, paths, and medical knowledge.

        ### Task Requirements:
        1. **Evidence Synthesis**:
           - **When path evidence exists**:
             * Clearly explain how it supports or refutes reasoning steps.
             * Highlight strong path-to-reasoning alignments.
           - **When path evidence is missing**:
             * Use medical knowledge to assess plausibility.
             * Note critical gaps or indirect support from other evidence.
           - Identify any misleading or irrelevant paths and discard them.

        2. **Reasoning Chain Evaluation**:
           - Rank reasoning chains based on their evidence strength:
             * High: Strong alignment with paths and medical knowledge.
             * Medium: Plausible but lacking direct path evidence.
             * Low: Weak support or contradicting evidence.
           - Clearly prioritize high-confidence chains in the final analysis.

        3. **Focus on Decision-Critical Insights**:
           - Highlight the most likely diagnosis based on combined evidence.
           - Provide concise reasons for deprioritizing less likely diagnoses.
           - Avoid overemphasizing diagnoses with weak or irrelevant support.

        4. **Address Uncertainty**:
           - Explicitly mention uncertainties in evidence or reasoning.
           - Adjust confidence levels accordingly.

        ### Output Format:
        Provide your response in the following JSON format:
        {
          "enhanced_information": "Your concise, evidence-based summary of the most likely diagnosis and key supporting insights."
        }

        ### Key Guidelines:
        - Focus on clinically significant findings and relationships.
        - Prioritize strong path evidence but do not ignore gaps.
        - Use medical knowledge to bridge evidence gaps.
        - Be concise but complete, ensuring the summary aids decision-making.
        - Avoid overcomplicating the explanation or including irrelevant details.
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

        if len(question.enhanced_graph.paths) == 0:
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

        if len(question.enhanced_graph.paths) == 0:
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
            Your task: Analyze how retrieved graph paths validate each step in the reasoning chains and use this validation to determine the most reliable answer. Since these paths were specifically retrieved based on the reasoning chains, each path potentially validates a specific reasoning step.

            ### Question
            {question.question}

            ### Options
            """
            for key, text in question.options.items():
                prompt += f"{key}. {text}\n"

            if question.reasoning_chain:
                prompt += "\n### Reasoning Chains for Validation:\n"
                for chain in question.reasoning_chain:
                    prompt += f"- {chain}\n"

            if question.enhanced_graph and question.enhanced_graph.paths:
                prompt += "\n### Retrieved Validation Paths:\n"
                for path in question.enhanced_graph.paths:
                    prompt += f"- {path}\n"

            prompt += """
            ### Task Requirements

            1. **Step-by-Step Validation**:
               - For each reasoning step (transition between states), find supporting paths
               - Example Analysis:
                 * Reasoning step: "Insulin" -> "decreased blood glucose"
                 * Supporting path: "(Insulin)-DECREASES->(Blood glucose)"
               - Identify which specific parts of paths validate which specific reasoning steps
               - Note when a path provides strong, partial, or no support for a step

            2. **Chain Strength Assessment**:
               - Evaluate reasoning chains based on path support at each step
               - Consider a chain stronger when:
                 * More steps are supported by paths
                 * Supporting paths are direct and clear
                 * Multiple paths support the same step
               - Identify which chains have the most comprehensive path support

            3. **Medical Knowledge Role**:
               - Use medical knowledge to interpret path support
               - When paths support a step, use medical knowledge to confirm the logic
               - When paths partially support a step, use medical knowledge to assess plausibility
               - Medical knowledge should complement, not override, path evidence

            4. **Decision Making**:
               - Base decision primarily on chains with strong path validation
               - Higher confidence when:
                 * Multiple paths support critical steps
                 * Path support aligns with medical knowledge
                 * Supported steps form a complete logical chain
               - Lower confidence when:
                 * Key steps lack path support
                 * Paths provide only partial validation
                 * Medical knowledge suggests alternative interpretations

            ### Output Format
            Provide your response in valid JSON format:
            {

                "chain_analysis": "Assessment of which chains have strongest path support",
                "final_analysis": "How path validation leads to the answer",
                "answer": "Option key (e.g., opa, opb, etc. only one and must one)",
                "confidence": Score between 0-100 based on path support strength
            }

            ### Key Guidelines
            - Focus on matching paths to specific reasoning steps
            - Evaluate support for each transition in the chains
            - Consider path support as primary evidence
            - Use medical knowledge to interpret, not override, path support
            - Confidence should reflect how well paths validate the crucial steps

            Remember:
            - Each path was retrieved to validate specific reasoning steps
            - Strong validation requires clear path-to-step correspondence
            - Multiple paths supporting a step increases confidence
            - Gaps in path support should lower confidence
            """
        try:
            messages = [
                {"role": "system", "content": "You are a medical expert using chain-of-thought reasoning."},
                {"role": "user", "content": prompt}
            ]

            result = self._get_completion(messages, "chain_of_thought", question)
            result = self._get_completion(messages, "final_answer", question)
            try:
                question.confidence = float(result.get("confidence", 0.0))
            except (TypeError, ValueError):
                question.confidence = 0.0

        except Exception as e:
            logging.error(f"Error in chain-of-thought answer: {str(e)}")
            question.chain_of_thought = ""
            _update_question_with_error(question)

