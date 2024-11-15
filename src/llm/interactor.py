from typing import Dict, List, Optional
import logging
import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from src.modules.MedicalQuestion import MedicalQuestion
from config import config


class LLMProcessor:
    """处理与LLM的所有交互"""

    def __init__(self):
        """使用全局配置初始化处理器"""
        self.client = OpenAI(api_key=config.openai["api_key"])
        self.model = config.openai.get("model", "gpt-4o-mini")
        self.temperature = config.openai.get("temperature", 0.3)
        self.logger = config.get_logger("llm_interaction")
        self.logger.info(f"Initialized LLM interaction with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        after=lambda f: logging.getLogger("casual_graphrag.llm_interaction").warning(
            f"Retrying API call after failure: {f.exception()}")
    )
    def _get_completion(self, messages: List[Dict], temperature: Optional[float] = None) -> str:
        """获取LLM回复"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}", exc_info=True)
            raise

    def direct_answer(self, question: MedicalQuestion) -> None:
        """直接回答问题，不使用任何额外知识"""
        prompt = f"""As a medical expert, please help answer this multiple choice question:

Question: {question.question}

Options:
"""
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        prompt += """
Please analyze and select the most appropriate answer. Respond in the following format:

Analysis: (Your medical reasoning)
Answer: (Option letter,only opa,opb,opc or opd, no addtional information)
Confidence: (A number from 0-100 indicating your confidence)
"""

        try:
            messages = [
                {"role": "system", "content": "You are a medical expert helping to answer multiple choice questions."},
                {"role": "user", "content": prompt}
            ]

            response = self._get_completion(messages)

            # Parse response and update question object
            for line in response.split('\n'):
                if line.startswith('Analysis:'):
                    question.reasoning = line.replace('Analysis:', '').strip()
                elif line.startswith('Answer:'):
                    question.answer = line.replace('Answer:', '').strip().lower()
                elif line.startswith('Confidence:'):
                    try:
                        question.confidence = float(line.replace('Confidence:', '').strip().rstrip('%'))
                    except ValueError:
                        self.logger.warning("Failed to parse confidence value")
                        question.confidence = 0

        except Exception as e:
            self.logger.error(f"Error in direct answer: {str(e)}", exc_info=True)

    def causal_enhanced_answer(self, question: MedicalQuestion) -> None:
        """利用因果路径增强的回答"""
        prompt = f"""As a medical expert, please help answer this multiple choice question using the provided causal relationships:

Question: {question.question}

Options:
"""
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        if question.casual_paths:
            prompt += "\nCausal relationships for each option:\n"
            for option in ['opa', 'opb', 'opc', 'opd']:
                if question.casual_paths.get(option):
                    prompt += f"\nOption {option} related causal paths:\n"
                    for path in question.casual_paths[option]:
                        prompt += f"- {path}\n"

        prompt += """
Based on the question, options, and causal relationships, please provide:

Analysis: (Explain your reasoning using the causal relationships)
Answer: (Option letter,only opa,opb,opc or opd, no addtional information)
Confidence: (A number from 0-100)
"""

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert analyzing questions using causal relationships."},
                {"role": "user", "content": prompt}
            ]

            response = self._get_completion(messages)

            for line in response.split('\n'):
                if line.startswith('Analysis:'):
                    question.reasoning = line.replace('Analysis:', '').strip()
                elif line.startswith('Answer:'):
                    question.answer = line.replace('Answer:', '').strip().lower()
                elif line.startswith('Confidence:'):
                    try:
                        question.confidence = float(line.replace('Confidence:', '').strip().rstrip('%'))
                    except ValueError:
                        question.confidence = 0

        except Exception as e:
            self.logger.error(f"Error in causal enhanced answer: {str(e)}", exc_info=True)

    def generate_reasoning_chain(self, question: MedicalQuestion) -> None:
        """基于因果路径生成推理链和需要验证的实体对"""
        self.logger.debug(f"Starting reasoning chain analysis")

        prompt = f"""As a medical expert, analyze this multiple-choice question to identify key medical relationships:

    Question: {question.question}

    Options:
    """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        if question.casual_paths:
            prompt += "\nCausal relationships for each option:\n"
            for option in ['opa', 'opb', 'opc', 'opd']:
                if question.casual_paths.get(option):
                    prompt += f"\nOption {option} related causal paths:\n"
                    for path in question.casual_paths[option]:
                        prompt += f"- {path}\n"

        prompt += """
        Based on the medical question, provided options, and known causal pathways, please provide:

        1. A step-by-step reasoning chain for answering this question. The reasoning chain must be correct, even if incomplete. You do not need to use all the causal pathways; include them only if they are relevant. Missing parts can be supplemented by retrieving entity pairs.

        2. Key medical entity pairs, where 'start' and 'end' correspond one-to-one, whose relationships and pathways, when retrieved, can help complete the reasoning chain and significantly aid in answering the question. Try to align these entities with professional UMLS entities but do not force them to be present in the UMLS knowledge graph.

        Format your response EXACTLY as follows:

        {
            "reasoning_chain": [
                "Step 1: First analyze...",
                "Step 2: Then consider...",
                "Step 3: Finally evaluate..."
            ],
            "entity_pairs_to_retrieve": [
                {
                    "start": ["start1, "start2"],
                    "end": ["end1", "end2"],
                    "reasoning": "Explanation of how this entity pair is relevant and helps complete the reasoning chain and answer the question" for each pair
                }
                // Ensuring 'start' and 'end' correspond one-to-one
            ]
        }

        IMPORTANT:
        - Use precise medical terms.
        - Provide clear, logical steps.
        - The reasoning chain must be correct, even if incomplete.
        - Response must be valid JSON.
        - Each entity pair should include main and alternative terms.
        - Do not force the entities to be present in the UMLS knowledge graph but try to align them with professional UMLS entities.
        """

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert analyzing questions. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ]

            response = self._get_completion(messages)
            self.logger.debug(f"Raw LLM response: {response}")

            # Clean up response to ensure valid JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]

            # Try to find JSON content within the response
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    response = response[start_idx:end_idx]

                result = json.loads(response)

                # Validate response structure
                if "reasoning_chain" not in result or "entity_pairs_to_retrieve" not in result:
                    raise ValueError("Missing required fields in response")

                # Update entity pairs
                if result["entity_pairs_to_retrieve"]:
                    start_entities = []
                    end_entities = []
                    for pair in result["entity_pairs_to_retrieve"]:
                        if isinstance(pair, dict) and "start" in pair and "end" in pair:
                            if isinstance(pair["start"], list) and pair["start"]:
                                start_entities.append(pair["start"][0])
                            if isinstance(pair["end"], list) and pair["end"]:
                                end_entities.append(pair["end"][0])

                    question.entities_original_pairs = {
                        'start': start_entities,
                        'end': end_entities
                    }

                # Update reasoning chain
                if result["reasoning_chain"]:
                    question.reasoning = "\n".join(result["reasoning_chain"])

                self.logger.info(
                    f"Successfully processed reasoning chain with {len(result['entity_pairs_to_retrieve'])} entity pairs")

            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing error: {str(e)}\nResponse: {response}")
                # 设置默认值
                question.entities_original_pairs = {'start': [], 'end': []}
                question.reasoning = "Error processing reasoning chain"

            except Exception as e:
                self.logger.error(f"Error processing reasoning chain: {str(e)}")
                question.entities_original_pairs = {'start': [], 'end': []}
                question.reasoning = "Error processing reasoning chain"

        except Exception as e:
            self.logger.error(f"Error in API call: {str(e)}")
            question.entities_original_pairs = {'start': [], 'end': []}
            question.reasoning = "Error in API call"

    def final_answer_with_all_paths(self, question: MedicalQuestion) -> None:
        """基于所有信息（因果路径、思维链、KG路径）生成最终答案"""
        prompt = f"""As a medical expert, please analyze this question using all available information:

Question: {question.question}

Options:
"""
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        # Add causal paths
        if question.casual_paths:
            prompt += "\nCausal relationships for each option:\n"
            for option in ['opa', 'opb', 'opc', 'opd']:
                if question.casual_paths.get(option):
                    prompt += f"\nOption {option} related causal paths:\n"
                    for path in question.casual_paths[option]:
                        prompt += f"- {path}\n"

        # Add reasoning chain
        if question.reasoning:
            prompt += f"\nReasoning process:\n{question.reasoning}\n"

        # Add KG paths
        if question.KG_paths:
            prompt += "\nVerified knowledge graph relationships:\n"
            for path in question.KG_paths:
                prompt += f"- {path}\n"

        prompt += """
Based on all the above information, please provide:

Final Analysis: (Synthesize all available information)
Answer: (Option letter,only opa,opb,opc or opd, no addtional information)
Confidence: (A number from 0-100)
"""

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert making decisions based on comprehensive evidence."},
                {"role": "user", "content": prompt}
            ]

            response = self._get_completion(messages)

            for line in response.split('\n'):
                if line.startswith('Final Analysis:'):
                    question.reasoning = line.replace('Final Analysis:', '').strip()
                elif line.startswith('Answer:'):
                    question.answer = line.replace('Answer:', '').strip().lower()
                elif line.startswith('Confidence:'):
                    try:
                        question.confidence = float(line.replace('Confidence:', '').strip().rstrip('%'))
                    except ValueError:
                        question.confidence = 0

        except Exception as e:
            self.logger.error(f"Error in final answer generation: {str(e)}", exc_info=True)