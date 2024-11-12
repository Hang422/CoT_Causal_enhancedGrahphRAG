from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import logging
from src.modules.data_format import (
    Question, QuestionGraphResult, QuestionAnalysis,
    FinalReasoning, EntityPairs, OptionKey, Answer
)
from config import config


class LLMInteraction:
    """Handles all LLM interactions for the medical QA system"""

    def __init__(self):
        """Initialize using configuration"""
        self.client = OpenAI(api_key=config.openai["api_key"])
        self.model = config.openai.get("model", "gpt-4o-mini")
        self.temperature = config.openai.get("temperature", 0.3)

        # Get logger for this module
        self.logger = config.get_logger("llm_interaction")
        self.logger.info(f"Initialized LLM interaction with model: {self.model}")

        # Cache path for storing responses
        self.cache_dir = config.paths["cache"] / "llm_responses"
        self.cache_dir.mkdir(exist_ok=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        after=lambda f: logging.getLogger("casual_graphrag.llm_interaction").warning(
            f"Retrying API call after failure: {f.exception()}")
    )
    def _get_completion(self, messages: List[Dict], temperature: Optional[float] = None) -> str:
        """Get completion from OpenAI API"""
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

    def get_causal_enhanced_answer(
            self,
            question: Question,
            graph_result: QuestionGraphResult
    ) -> Answer:
        """Get answer using causal path enhancement"""
        self.logger.debug(f"Processing causal enhanced answer for question ID: {id(question)}")

        # Construct prompt
        prompt = f"""As a medical expert, please help answer this multiple choice question based on the following information:

Question: {question.text}

Options:
"""
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        # Add causal paths
        prompt += "\nRelevant causal relationships from medical knowledge graph:\n"

        # Add question paths
        for path in graph_result.question_paths:
            path_str = " -> ".join([
                                       f"{node_name} [{rel}]"
                                       for node_name, rel in zip(path.node_names[:-1], path.relationships)
                                   ] + [path.node_names[-1]])
            prompt += f"- {path_str}\n"

        # Add option paths
        for opt, paths in graph_result.option_paths.items():
            if paths:
                prompt += f"\nPaths related to option {opt}:\n"
                for path in paths:
                    path_str = " -> ".join([
                                               f"{node_name} [{rel}]"
                                               for node_name, rel in zip(path.node_names[:-1], path.relationships)
                                           ] + [path.node_names[-1]])
                    prompt += f"- {path_str}\n"

        prompt += """
Based on the question, options, and causal relationships provided above, please analyze and select the most appropriate answer. Respond in the following format:

Analysis: (Detailed explanation of your reasoning process)
Answer: (Option letter)
Confidence: (A number from 0-100 indicating your confidence)
"""

        try:
            # Get LLM response
            messages = [
                {"role": "system", "content": "You are a medical expert helping to answer multiple choice questions."},
                {"role": "user", "content": prompt}
            ]

            response = self._get_completion(messages)

            # Parse response
            analysis = ""
            answer_letter = None
            confidence = 0

            for line in response.split('\n'):
                if line.startswith('Analysis:'):
                    analysis = line.replace('Analysis:', '').strip()
                elif line.startswith('Answer:'):
                    answer_text = line.replace('Answer:', '').strip().lower()
                    try:
                        answer_letter = OptionKey(answer_text)
                    except ValueError:
                        self.logger.warning(f"Invalid answer option: {answer_text}")
                        continue
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.replace('Confidence:', '').strip().rstrip('%'))
                    except ValueError:
                        self.logger.warning("Failed to parse confidence value")
                        confidence = 0

            if not answer_letter:
                raise ValueError("No valid answer option found in response")

            answer = Answer(
                question=question,
                answer=answer_letter,
                confidence=confidence,
                explanation=analysis,
                isCorrect=(answer_letter == question.correct_answer)
            )

            self.logger.info(f"Generated answer for question ID {id(question)} with confidence: {confidence}")
            return answer

        except Exception as e:
            self.logger.error(f"Error generating causal enhanced answer: {str(e)}", exc_info=True)
            raise

    def get_reasoning_chain(
            self,
            question: Question,
            graph_result: QuestionGraphResult
    ) -> QuestionAnalysis:
        """Get initial reasoning chain and entity pairs to verify"""
        self.logger.debug(f"Starting reasoning chain analysis for question ID: {id(question)}")

        # Construct prompt
        prompt = f"""As a medical expert, help analyze this multiple-choice question to identify key medical concepts and relationships that need further investigation to understand the correct answer.

    Question: {question.text}

    Options:
    """
        for key, text in question.options.items():
            prompt += f"{key.value}. {text}\n"

        # Add known paths if available
        prompt += "\nKnown medical relationships (these may be incomplete or contain noise):\n"
        for path in graph_result.question_paths:
            path_str = " -> ".join([
                                       f"{node_name} [{rel}]"
                                       for node_name, rel in zip(path.node_names[:-1], path.relationships)
                                   ] + [path.node_names[-1]])
            prompt += f"- {path_str}\n"

        prompt += """
    Your tasks are:
    1. Break down the medical reasoning process into steps to understand the question and options.
    2. Based on the reasoning, identify the key medical entity pairs (start and end points) whose relationships need to be investigated further to help answer the question.

    Please focus on:
    - What medical concepts are crucial to understand in this question?
    - Which relationships between these concepts, if verified, would be most helpful in determining the correct answer?
    - What additional entity pairs should we consider retrieving paths for, to aid in answering the question?

    IMPORTANT: Return ONLY a valid JSON object following exactly this format:
    {
        "reasoning_chain": [
            "step1: ...",
            "step2: ...",
            "step3: ..."
        ],
        "entity_pairs_to_retrieve": [
            {
                "start": ["start entity term", "alternative term"],
                "end": ["end entity term", "alternative term"],
                "reasoning": "Why investigating the relationship between these entities is crucial for answering the question."
            }
        ]
    }

    Guidelines for content:
    - The reasoning chain should outline clear steps in understanding the medical concepts and reasoning needed to answer the question.
    - For each entity pair, explain why investigating the relationship between these entities would help in determining the correct answer.
    - Include alternative medical terms or synonyms for entities.
    - Focus on the most critical entity pairs that would aid in answering the question.
    - Do not attempt to determine the answer yet.
    - Return the JSON response directly without any additional text or formatting.
    """

        try:
            # Get LLM response
            messages = [
                {"role": "system",
                 "content": "You are a medical expert analyzing questions using verified causal relationships. Always return your response as a pure JSON object without any markdown formatting or additional text."},
                {"role": "user", "content": prompt}
            ]

            response = self._get_completion(messages)

            try:
                # Try to clean the response if needed
                response = response.strip()
                if response.startswith('```json'):
                    response = response[7:]
                if response.endswith('```'):
                    response = response[:-3]

                result = json.loads(response)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                raise ValueError("Invalid JSON response from LLM")

            # Validate required fields
            if "reasoning_chain" not in result or "entity_pairs_to_retrieve" not in result:
                raise ValueError("Missing required fields in LLM response")

            entity_pairs = [
                EntityPairs(
                    start=pair['start'],
                    end=pair['end'],
                    reasoning=pair['reasoning']
                )
                for pair in result.get('entity_pairs_to_retrieve', [])
            ]

            analysis = QuestionAnalysis(
                reasoning_chain=result.get('reasoning_chain', []),
                entity_pairs=entity_pairs
            )

            self.logger.info(
                f"Generated reasoning chain for question ID {id(question)} "
                f"with {len(entity_pairs)} entity pairs and {len(analysis.reasoning_chain)} steps"
            )
            return analysis

        except Exception as e:
            self.logger.error(f"Error in reasoning chain generation: {str(e)}", exc_info=True)
            raise

    def get_final_answer(
            self,
            reasoning: FinalReasoning
    ) -> Answer:
        """Get final answer using all accumulated information"""
        self.logger.debug(f"Generating final answer for question ID: {id(reasoning.question)}")

        # Construct prompt
        prompt = f"""As a medical expert, please analyze this question using all provided information:

Question: {reasoning.question.text}

Options:
"""
        for key, text in reasoning.question.options.items():
            prompt += f"{key}. {text}\n"

        prompt += "\nInitial reasoning chain:\n"
        for step in reasoning.reasoning_chain:
            prompt += f"- {step}\n"

        # Add verified entity pairs and their paths
        prompt += "\nVerified relationships between key entities:\n"
        for pair, paths in reasoning.entity_pairs:
            prompt += f"\nRelationship between {pair.start} and {pair.end}:\n"
            for path in paths:
                path_str = " -> ".join([
                                           f"{node_name} [{rel}]"
                                           for node_name, rel in zip(path.node_names[:-1], path.relationships)
                                       ] + [path.node_names[-1]])
                prompt += f"- {path_str}\n"

        prompt += """
Based on all this information, please provide:

Final Analysis: (Explain how the verified relationships inform your decision)
Answer: (Option letter)
Confidence: (A number from 0-100)
"""

        try:
            # Get LLM response
            messages = [
                {"role": "system",
                 "content": "You are a medical expert making a final decision based on verified evidence."},
                {"role": "user", "content": prompt}
            ]

            response = self._get_completion(messages)

            # Parse response
            analysis = ""
            answer_letter = None
            confidence = 0

            for line in response.split('\n'):
                if line.startswith('Final Analysis:'):
                    analysis = line.replace('Final Analysis:', '').strip()
                elif line.startswith('Answer:'):
                    answer_text = line.replace('Answer:', '').strip().lower()
                    try:
                        answer_letter = OptionKey(answer_text)
                    except ValueError:
                        self.logger.warning(f"Invalid final answer option: {answer_text}")
                        continue
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.replace('Confidence:', '').strip().rstrip('%'))
                    except ValueError:
                        self.logger.warning("Failed to parse final confidence value")
                        confidence = 0

            if not answer_letter:
                raise ValueError("No valid answer option found in final response")

            answer = Answer(
                question=reasoning.question,
                answer=answer_letter,
                confidence=confidence,
                explanation=analysis,
                isCorrect=(answer_letter == reasoning.question.correct_answer)
            )

            self.logger.info(
                f"Generated final answer for question ID {id(reasoning.question)} "
                f"with confidence: {confidence}"
            )
            return answer

        except Exception as e:
            self.logger.error(f"Error generating final answer: {str(e)}", exc_info=True)
            raise

    def get_direct_answer(
            self,
            question: Question
    ) -> Answer:
        """Get answer directly without any additional knowledge or reasoning

        Args:
            question: Question object containing text and options

        Returns:
            Answer object containing the selected option and confidence
        """
        self.logger.debug(f"Getting direct answer for question ID: {id(question)}")

        # Construct simple prompt
        prompt = f"""As a medical expert, please answer this multiple choice question:

Question: {question.text}

Options:
"""
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        prompt += """
Please analyze and select the most appropriate answer. Respond in the following format:

Analysis: (Your medical reasoning for this answer)
Answer: (Option letter)
Confidence: (A number from 0-100 indicating your confidence)
"""

        try:
            messages = [
                {"role": "system", "content": "You are a medical expert helping to answer multiple choice questions."},
                {"role": "user", "content": prompt}
            ]

            response = self._get_completion(messages)

            # Parse response
            analysis = ""
            answer_letter = None
            confidence = 0

            for line in response.split('\n'):
                if line.startswith('Analysis:'):
                    analysis = line.replace('Analysis:', '').strip()
                elif line.startswith('Answer:'):
                    answer_text = line.replace('Answer:', '').strip().lower()
                    try:
                        answer_letter = OptionKey(answer_text)
                    except ValueError:
                        self.logger.warning(f"Invalid answer option: {answer_text}")
                        continue
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.replace('Confidence:', '').strip().rstrip('%'))
                    except ValueError:
                        self.logger.warning("Failed to parse confidence value")
                        confidence = 0

            if not answer_letter:
                raise ValueError("No valid answer option found in response")

            answer = Answer(
                question=question,
                answer=answer_letter,
                confidence=confidence,
                explanation=analysis,
                isCorrect=(answer_letter == question.correct_answer)
            )

            self.logger.info(f"Generated direct answer for question ID {id(question)} with confidence: {confidence}")
            return answer

        except Exception as e:
            self.logger.error(f"Error generating direct answer: {str(e)}", exc_info=True)
            raise
