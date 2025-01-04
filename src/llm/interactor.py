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
import re

def format_reasoning_chains(raw_text: str) -> List[str]:
    """
    将非标准格式的思维链转换为标准格式

    Args:
        raw_text: 原始响应文本

    Returns:
        List[str]: 标准格式的思维链列表
    """

    def clean_step(step: str) -> str:
        """清理单个步骤"""
        # 移除数字编号
        step = re.sub(r'^\d+\.\s*', '', step)
        # 移除多余空格
        step = ' '.join(step.split())
        # 移除"above side effect"等重复短语
        step = re.sub(r'\s*->\s*(?:above\s+)?side\s+effect(?:\s*$)?', '', step)
        return step.strip()

    def extract_confidence(text: str) -> Optional[str]:
        """提取置信度"""
        confidence_patterns = [
            r'(?:Confidence:?\s*)?(\d+)%',
            r'confidence\s*(?:is|:)\s*(\d+)',
            r'->?\s*(\d+)%'
        ]

        for pattern in confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)}%"
        return None

    def process_single_chain(chain_text: str) -> Optional[str]:
        """处理单个思维链"""
        # 移除常见的标题/前缀
        chain_text = re.sub(r'^.*(?:Reasoning Chain|Chain)\s*\d*:?\s*', '', chain_text, flags=re.IGNORECASE)

        # 提取置信度
        confidence = extract_confidence(chain_text)
        if not confidence:
            return None

        # 移除置信度部分以处理步骤
        chain_text = re.sub(r'(?:Confidence:?\s*)?(?:\d+%|confidence\s*(?:is|:)\s*\d+).*$', '', chain_text,
                            flags=re.IGNORECASE)

        # 分割步骤
        if '->' in chain_text:
            steps = [clean_step(step) for step in chain_text.split('->')]
        else:
            steps = [clean_step(step) for step in chain_text.split('\n') if step.strip()]

        # 过滤空步骤并组合
        steps = [step for step in steps if step]
        if not steps:
            return None

        # 构建标准格式
        return f"CHAIN: {' -> '.join(steps)} -> {confidence}"

    # 主处理逻辑
    chains = []

    # 1. 尝试按编号或分隔符分割多个链
    chain_separators = [
        r'(?:###?\s*)?(?:Reasoning\s+)?Chain\s*\d+:',
        r'^\d+\.\s',
        r'\n\s*\n'
    ]

    text_chunks = []
    for separator in chain_separators:
        if re.search(separator, raw_text, re.MULTILINE | re.IGNORECASE):
            text_chunks = [chunk.strip() for chunk in re.split(separator, raw_text) if chunk.strip()]
            break

    if not text_chunks:
        text_chunks = [raw_text]

    # 2. 处理每个潜在的链
    for chunk in text_chunks:
        formatted_chain = process_single_chain(chunk)
        if formatted_chain:
            chains.append(formatted_chain)

    return chains


def fix_reasoning_chains(response_text: str) -> List[str]:
    """
    修复并标准化推理链格式

    Args:
        response_text: GPT的原始响应文本

    Returns:
        List[str]: 修复后的标准格式推理链列表
    """
    try:
        return format_reasoning_chains(response_text)
    except Exception as e:
        logging.error(f"Error formatting reasoning chains: {str(e)}")
        return []


# 在LLMProcessor中使用

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

    def __init__(self, log_path:str):
        """Initialize logger with configured paths and interaction types"""
        self.base_log_dir = config.paths["logs"] / "llm_logs" / log_path
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        self.error_dir = self.base_log_dir / "errors"
        self.error_dir.mkdir(parents=True, exist_ok=True)

        # Model-specific logging directories
        self.models = {
            'gpt-3.5-turbo': 'gpt-3.5-turbo',
            'gpt-4-turbo': 'gpt-4-turbo',
            'gpt-4o': 'gpt-4o',
            'gpt-4o-mini': 'gpt-4o-mini'
        }

        # Interaction types
        # Add this to the interaction_types dictionary in LLMLogger.__init__
        self.interaction_types = {
            'direct_answer': 'direct_answer',
            'reasoning_chain': 'reasoning_chain',
            'enhancement_with_chain_complete_with_chain': 'enhancement_with_chain_complete_with_chain',
            'enhancement_with_chain_kg_only': 'enhancement_with_chain_kg_only',
            'enhancement_with_chain_without_enhancer': 'enhancement_with_chain_without_enhancer',
            'answer_with_enhancement_causal_only': 'answer_with_enhancement_causal_only',
            'answer_with_enhancement_kg_only': 'answer_with_enhancement_kg_only',
            'answer_with_enhancement_without_enhancer': 'answer_with_enhancement_without_enhancer',  # Add this line
            'answer_with_enhancement_complete_with_chain': 'answer_with_enhancement_complete_with_chain',
            'answer_with_CoT': 'answer_with_CoT',
            'answer_normal_rag': 'answer_normal_rag'
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

    def __init__(self, log_path: str):
        self.client = OpenAI(api_key=config.openai["api_key"])
        self.model = config.openai.get("model")
        self.model_name = config.openai.get("model")
        self.temperature = config.openai.get("temperature")
        self.logger = LLMLogger(log_path)
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
        prompt = f"""
        You are a medical expert. Generate multiple reasoning chains for the following question.

            Each chain:
            - Uses '->' to connect sequential reasoning states.
            - Each step is a single state or result, not a complete causal sentence in one step.
              Example:
                Correct: "Insulin" -> "decreased blood glucose" -> "improved metabolism" -> 90%
                Incorrect: "Insulin decreases blood glucose" in one step.
            - End with a confidence percentage (0-100%).

            If a step clearly conflicts with widely accepted medical knowledge, you may indicate uncertainty (e.g., 
            "possible", "unclear") and slightly lower confidence. However, do this only if necessary. If unsure, 
            it is acceptable to show a plausible chain with moderate confidence.

            ### Question
            {question.question}

            ### Options
    """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"
        prompt += """
            ### Instructions:
            - Keep steps simple, each step just one state or outcome.
            - '->' separates each step.
            
            ### Output Format:
            Start each chain with "CHAIN:" on a new line.
            Example:
            CHAIN: "Insulin" -> "decreased blood glucose" -> "improved metabolic state" -> 90%"""

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

            chains = fix_reasoning_chains(response_text)

            # 更新问题的推理链
            question.reasoning_chain = chains if chains else []

            if not chains:
                logging.warning(f"No valid reasoning chains found for question: {question.question}")
                logging.warning(f"Raw response: {response_text}")

        except Exception as e:
            logging.error(f"Error in reasoning chain generation: {str(e)}", exc_info=True)
            question.reasoning_chain = []

    def enhance_information(self, question: MedicalQuestion, flag:str) -> None:
        """
        Enhancement with standardized processing:
        1. We have an 'enhanced_graph' containing multiple evidence paths (strings).
        2. We want to prune/merge these paths so that only the medically relevant and
           question-focused info is retained.
        3. Then produce a concise "enhanced_information" summary that aligns with standard medical knowledge
           and truly helps in answering the question.

        If no paths are available, skip.
        """

        if not question.enhanced_graph.paths:
            return

        # 构建角色设定
        if question.topic_name:
            system_prompt = f"You are a medical expert specializing in {question.topic_name}."
        else:
            system_prompt = "You are a medical expert."

        # 构建用户提示
        user_prompt = f"""
    You have retrieved multiple evidence paths (see below) from a knowledge graph search. 
    They may be correct in their own right but can be broad or tangential to the question. 
    Your goal is to integrate the relevant portions with standard medical knowledge, 
    and produce a short summary ('enhanced_information') that helps answer the question accurately.

    ## Question
    {question.question}

    ## Options
    """
        # 列出选项
        for key, text in question.options.items():
            user_prompt += f"{key}. {text}\n"

        # 列出增强图路径
        user_prompt += "\n## Retrieved Evidence Paths\n"
        for path_str in question.enhanced_graph.paths:
            user_prompt += f"{path_str}\n"

        # 具体指令
        user_prompt += """
    ## Instructions
    1. Review each evidence path: If it contradicts well-established medical facts, ignore/flag it.
    2. If certain paths are irrelevant to the question, ignore them.
    3. Summarize only the key, correct, and question-relevant pieces of info.
    4. If some aspects are uncertain, note them briefly, but still focus on a consensus-based conclusion.
    5. Output the final result as valid JSON:
    {
      "enhanced_information": "Your short, clinically relevant summary that uses the most pertinent evidence and standard knowledge."
    }
    """

        # 组装
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            result = self._get_completion(messages, "enhancement" + flag, question)
            question.enhanced_information = result.get("enhanced_information", "")
        except Exception as e:
            logging.error(f"Error in enhancement: {str(e)}")
            question.enhanced_information = ""

    def enhance_information_with_chain(self, question: MedicalQuestion, flag:str) -> None:
        """Enhancement with standardized processing:
           Use standard medical knowledge and retrieved evidence paths to prune and verify the reasoning chains.
           The retrieved paths might be broad but are correct; do not be misled by irrelevant details.
           After verification, produce a concise, consensus-aligned enhanced_information."""

        # 如果没有增强图路径，无需处理
        if len(question.enhanced_graph.paths) == 0:
            return

        if question.topic_name is not None:
            prompt = f"You are a medical expert specializing in {question.topic_name}."
        else:
            prompt = "You are a medical expert."

        prompt += f"""
        Your task:
        - You have per-option reasoning chains (from previous step).
        - You have retrieved evidence paths (enhanced graph info) that are correct but may be broad and not directly addressing the key point.
        - Use standard medical/biochemical consensus to evaluate each chain.
        - If a chain step contradicts consensus, correct or remove it.
        - Check evidence paths: if they provide relevant support or clarify consensus, incorporate them.
        - If paths are too broad or not relevant, do not let them mislead you.
        - Aim to produce a final "enhanced_information" that:
          1. Reflects corrected, consensus-aligned reasoning for each option,
          2. Integrates helpful evidence from paths,
          3. Excludes misleading or irrelevant info,
          4. Focuses on what best helps answer the question correctly.

        ### Question
        {question.question}

        ### Options
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        if question.reasoning_chain:
            prompt += "\n### Reasoning Chains per Option:\n"
            for chain in question.reasoning_chain:
                prompt += f"{chain}\n"

        if question.enhanced_graph and len(question.enhanced_graph.paths) > 0:
            prompt += "\n### Retrieved Evidence Paths (broad but correct):\n"
            for path in question.enhanced_graph.paths:
                prompt += f"{path}\n"

        prompt += """
        ### Instructions:
        1. Recall standard consensus facts relevant to the question.
        2. For each option's chain, compare steps to consensus and paths:
           - Remove/adjust steps contradicting known facts.
           - If paths confirm or clarify a point aligned with consensus, use them.
           - Ignore irrelevant or overly broad paths that don't help.
        3. If uncertain, note uncertainty but choose the best consensus-supported interpretation.
        4. Output a short "enhanced_information" summarizing the corrected reasoning and relevant evidence that truly aids in final answer determination.

        ### Output Format:
        {
          "enhanced_information": "A concise, consensus-aligned summary integrating corrected reasoning and relevant evidence."
        }
        """

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert making decisions based on enhanced information and careful verification."},
                {"role": "user", "content": prompt}
            ]

            result = self._get_completion(messages, "enhancement_with_chain" + flag, question)
            question.enhanced_information = result.get("enhanced_information", "")

        except Exception as e:
            logging.error(f"Error in enhancement: {str(e)}")
            question.enhanced_information = ""

    def answer_with_enhanced_information(self, question: MedicalQuestion, flag: str) -> None:
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
                    You are a medical expert determining the final answer.

            ### Instructions:
            - Use standard medical knowledge as the primary guide.
            - Consider enhanced information only if it aligns with consensus.
            - If conflicting or unclear, present the most likely correct answer based on known facts.
            - It's acceptable to show moderate confidence if the question is complex, but still choose one best answer.


                    ### Question
                    {question.question}

                    ### Options
                    """
            for key, text in question.options.items():
                prompt += f"{key}. {text}\n"

            if question.enhanced_information:
                prompt += f"\n### Enhanced Information (For Contextual Support):\n{question.enhanced_information}\n"

            prompt += """
                    ### Task:
                1. Identify the core medical principle and the most likely correct option based on consensus.
                2. If the evidence is not perfectly clear, pick the best-supported option and explain the reasoning.
                3. Provide a final analysis and a confidence score (0-100%).

                ### Output Format:
                {
                  "final_analysis": "Step-by-step reasoning, prioritizing medical consensus and acknowledging complexity if present.",
                  "answer": "Option key (e.g. opa, opb, ...)",
                  "confidence": A number between 0-100
                }
                    """

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert making decisions based on enhanced information."},
                {"role": "user", "content": prompt}
            ]

            result = self._get_completion(messages, "answer_with_enhancement" + flag, question)
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
            You are a medical expert determining the final answer.

                ### Instructions:
                - Use standard medical knowledge as the primary guide.
                - Consider enhanced information only if it aligns with consensus.
                - If conflicting or unclear, present the most likely correct answer based on known facts.
                - It's acceptable to show moderate confidence if the question is complex, but still choose one best answer.
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

            if question.enhanced_graph and len(question.enhanced_graph.paths) > 0:
                prompt += "\n### Retrieved Validation Paths:\n"
                for path in question.enhanced_graph.paths:
                    prompt += f"- {path}\n"

            prompt += """
            ### Task:
                1. Identify the core medical principle and the most likely correct option based on consensus.
                2. If the evidence is not perfectly clear, pick the best-supported option and explain the reasoning.
                3. Provide a final analysis and a confidence score (0-100%).
                
                ### Output Format:
                {
                  "final_analysis": "Step-by-step reasoning, prioritizing medical consensus and acknowledging complexity if present.",
                  "answer": "Option key (e.g. opa, opb, ...)",
                  "confidence": A number between 0-100
                }
            """
        try:
            messages = [
                {"role": "system", "content": "You are a medical expert using chain-of-thought reasoning."},
                {"role": "user", "content": prompt}
            ]

            result = self._get_completion(messages, "answer_with_CoT", question)
            try:
                question.confidence = float(result.get("confidence", 0.0))
            except (TypeError, ValueError):
                question.confidence = 0.0

        except Exception as e:
            logging.error(f"Error in chain-of-thought answer: {str(e)}")
            question.chain_of_thought = ""
            _update_question_with_error(question)

    def answer_normal_rag(self, question: MedicalQuestion) -> None:
        """Answer question using vector search results"""
        if question.topic_name is not None:
            prompt = f"You are a medical expert specializing in the field of {question.topic_name}."
        else:
            prompt = "You are a medical expert."

        prompt += f"""Please help answer this {'multiple choice ' if question.is_multi_choice else ''}question:

    Question: {question.question}

    Options:
    """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        if question.normal_results:
            prompt += "\n### Retrieved Similar Medical Knowledge:\n"
            for result in question.normal_results:
                prompt += f"- {result}\n"

        prompt += """
    ### Instructions:
            - Use standard medical knowledge as the primary guide.
            - Consider enhanced information only if it aligns with consensus.
            - If conflicting or unclear, present the most likely correct answer based on known facts.
            - It's acceptable to show moderate confidence if the question is complex, but still choose one best answer.
    ### Task:
                1. Identify the core medical principle and the most likely correct option based on consensus.
                2. If the evidence is not perfectly clear, pick the best-supported option and explain the reasoning.
                3. Provide a final analysis and a confidence score (0-100%).
                
    ### Output Format
    Provide your response in valid JSON format:
    {   
        "final_analysis": "Your analysis of how the retrieved relationships support the answer",
        "answer": "Option key from the available options (only key, like opa)",
        "confidence": Score between 0-100 based on evidence strength and relevance
    }
    """

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert making decisions based on retrieved medical knowledge."},
                {"role": "user", "content": prompt}
            ]

            result = self._get_completion(messages, "answer_normal_rag", question)
            _update_question_from_result(question, result)

        except Exception as e:
            logging.error(f"Error in normal answer: {str(e)}")
            _update_question_with_error(question)


