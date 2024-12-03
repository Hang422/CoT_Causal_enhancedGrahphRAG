from typing import Dict, List, Optional
import logging
import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from src.modules.MedicalQuestion import MedicalQuestion
from config import config
from datetime import datetime
from pathlib import Path
import hashlib
from src.graphrag.query_processor import QueryProcessor

class GPTLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "gpt_logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_interaction(self, messages: list, response: str, interaction_type: str) -> None:
        """Log a single GPT interaction"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a unique identifier for this interaction
        interaction_id = hashlib.md5(
            f"{timestamp}_{interaction_type}".encode()
        ).hexdigest()[:8]

        # Create the log entry
        log_entry = {
            "timestamp": timestamp,
            "interaction_type": interaction_type,
            "messages": messages,
            "response": response
        }

        # Save to file
        log_file = self.output_dir / f"{timestamp}_{interaction_id}_{interaction_type}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

class LLMProcessor:
    """处理与LLM的所有交互"""

    def __init__(self):
        """使用全局配置初始化处理器"""
        self.client = OpenAI(api_key=config.openai["api_key"])
        self.model = config.openai.get("model")
        self.temperature = config.openai.get("temperature", 0.3)
        self.logger = config.get_logger("llm_interaction")
        self.logger.info(f"Initialized LLM interaction with model: {self.model}")
        # 初始化 GPT 日志记录器
        self.gpt_logger = GPTLogger(config.paths["output"])

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
            response_text = response.choices[0].message.content

            # 记录交互
            self.gpt_logger.log_interaction(
                messages=messages,
                response=response_text,
                interaction_type="completion"
            )

            return response_text
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}", exc_info=True)
            raise

    def direct_answer(self, question: MedicalQuestion) -> None:
        """直接回答问题，不使用任何额外知识"""
        prompt = f"""As a medical expert in the filed of {question.topic_name}, please help answer this multiple choice question:

Question: {question.question}

Options:
"""
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        prompt += """
Please analyze and select the most appropriate answer. Respond in the following format:

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
        prompt = f"""As a medical expert in the filed of {question.topic_name}, please help answer this multiple choice question using the provided causal relationships:

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

        # 构建初始 Prompt
        prompt = f"""You are a medical expert specializing in the field of {question.topic_name}. Your task involves two main objectives:

        1. **Generate Causal Chains (Causal Analysis)**: Analyze the provided causal relationships (if any) or infer them directly from the question and options. The goal is to identify direct relationships between entities (e.g., symptoms, organisms, anatomical structures, pathways) relevant to answering the question.

        2. **Infer Additional Entity Pairs**: Extend the reasoning chain by identifying additional causal entity pairs (start and end) that are required to form a complete reasoning chain but are not explicitly covered in the original pathways.

        ### Important Note on Provided Causal Relationships
        - The causal relationships provided in `question.casual_paths` are **preliminary and rough**. They may include irrelevant or incorrect information and are only intended as an initial reference.
        - If no causal relationships are provided or if they are unhelpful, infer causal entities directly from the question and options based on your domain knowledge and expertise.

        ### Question
        {question.question}

        ### Options
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        # Add causal paths if provided
        if question.casual_paths:
            prompt += "\n### Provided Causal Relationships for Each Option\n"
            for option in ['opa', 'opb', 'opc', 'opd']:
                if question.casual_paths.get(option):
                    prompt += f"\nOption {option} causal paths:\n"
                    for path in question.casual_paths[option]:
                        prompt += f"- {path}\n"
        else:
            prompt += "\nNo causal relationships are provided for this question."

        # Specify task and output format
        prompt += f"""

        ### Task
        Your task is to:
        1. **Causal Analysis**:
           - Extract and refine the causal relationships provided in the question (if any) by identifying `start` and `end` entities for each causal pathway. Ensure all entities align with UMLS standards and avoid vague modifiers like "Decreased" or "Inhibition."
           - If no causal pathways are provided, infer the most relevant causal entities based on the question and options.

        2. **Reasoning Chain and Additional Entity Pairs**:
           - Generate a clear and logical reasoning chain for answering the question. 
           - Convert the reasoning chain into structured causal relationships expressed as `start` and `end` pairs.
           - Identify additional causal entity pairs (start and end) that are necessary to complete the reasoning chain but are not part of the initial causal pathways.

        ### Output Format
        The output should be in the following JSON format:

        {{
            "causal_analysis": {{
                "start": ["Entity1", "Entity2", "Entity3"],
                "end": ["Entity4", "Entity5", "Entity6"]
            }},
            "additional_entity_pairs": {{
                "start": ["EntityA", "EntityB"],
                "end": ["EntityC", "EntityD"]
            }}
        }}

        ### Key Guidelines
        1. Ensure that the `start` and `end` arrays are **one-to-one aligned** and represent meaningful causal relationships.
        2. Use **precise UMLS-standardized terms** for all entities, avoiding abbreviations or vague modifiers.
        3. If no causal relationships are provided or useful, infer the causal entities based on medical knowledge and question context.
        4. The `causal_analysis` should focus on key causal chains relevant to answering the question, while `additional_entity_pairs` should extend the reasoning process.
        5. Avoid explanations or justifications in the output; stick to structured `start` and `end` pairs.
        6. Ensure terms can be directly queried in UMLS or similar medical knowledge graphs.
        """

        self.logger.debug(f"Prompt sent to LLM:\n{prompt}")

        # 调用 LLM 获取结果的代码保持不变

        try:
            # 调用 LLM 获取结果
            messages = [
                {"role": "system",
                 "content": "You are a medical expert analyzing questions. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            response = self._get_completion(messages)
            self.logger.debug(f"Raw LLM response: {response}")

            # 清理响应确保有效 JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]

            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                response = response[start_idx:end_idx]

            result = json.loads(response)

            # 验证响应结构
            if "additional_entity_pairs" not in result or "causal_analysis" not in result:
                raise ValueError("Missing required fields in response")

            # 提取因果分析结果
            if result.get("causal_analysis"):
                question.casual_paths_nodes_refine = {
                    'start': result["causal_analysis"].get("start", []),
                    'end': result["causal_analysis"].get("end", [])
                }
                self.logger.debug(f"Extracted causal analysis: {question.casual_paths_nodes_refine}")

            # 提取额外实体对
            if result.get("additional_entity_pairs"):
                question.entities_original_pairs = {
                    'start': result["additional_entity_pairs"].get("start", []),
                    'end': result["additional_entity_pairs"].get("end", [])
                }
                self.logger.debug(f"Extracted additional entity pairs: {question.entities_original_pairs}")

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}\nResponse: {response}")
            question.casual_paths_nodes_refine = {'start': [], 'end': []}
            question.entities_original_pairs = {'start': [], 'end': []}
            question.reasoning = "Error processing reasoning chain"

        except Exception as e:
            self.logger.error(f"Error processing reasoning chain: {str(e)}, Response: {response}")
            question.casual_paths_nodes_refine = {'start': [], 'end': []}
            question.entities_original_pairs = {'start': [], 'end': []}
            question.reasoning = "Error processing reasoning chain"

    def final_answer_with_all_paths(self, question: MedicalQuestion) -> None:
        """基于所有信息（因果路径、思维链、KG路径）生成最终答案"""
        prompt = f"""As a medical expert in the field of {question.topic_name}, please analyze this question using all available information:

Question: {question.question}

Options:
"""
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        # Add causal paths

        if question.CG_paths:
            prompt += "\nVerified casual graph relationships:\n"
            for path in question.CG_paths:
                prompt += f"- {path}\n"

        # Add reasoning chain
        if question.reasoning_chain:
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

    def answer_with_reasoning(self, question: MedicalQuestion) -> None:
        prompt = f"""As a medical expert, please analyze this question using all available information:

        Question: {question.question}

        Options:
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        # Add reasoning chain
        if question.reasoning:
            prompt += f"\nReasoning process:\n{question.reasoning}\n"

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

    def enhance_information(self, question: MedicalQuestion) -> None:
        """融合所有信息，生成增强后的信息"""
        self.logger.debug(f"Starting information enhancement")

        prompt = f"""You are a medical expert specializing in the field of {question.topic_name}. Your task is to integrate all the provided information, ensuring its truthfulness and relevance, enhance it by trimming irrelevant or misleading parts, and prepare it for answering the question.

        ### Question
        {question.question}

        ### Options
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        # 添加KG_paths（如果有）
        if question.KG_paths:
            prompt += "\n### Knowledge Graph Paths\n"
            for path in question.KG_paths:
                prompt += f"- {path}\n"

        # 添加CG_paths（如果有）
        if hasattr(question, 'CG_paths') and question.CG_paths:
            prompt += "\n### Causal Graph Paths\n"
            for path in question.CG_paths:
                prompt += f"- {path}\n"

        # 指定任务与输出格式
        prompt += """
        ### Task
        - Analyze and synthesize all the above information.
        - Verify the truthfulness and accuracy of the information.
        - **Trim irrelevant, redundant, or misleading parts.**
        - Merge and integrate relevant data from different sources coherently.
        - Ensure that the enhanced information is accurate, relevant, and logically supports the correct answer.
        - Prepare a concise and coherent summary that can be used to directly answer the question.
        - Do NOT provide the final answer to the question at this stage, nor should you make it obvious which option is correct.
        - **Ensure all information is accurate and based on established medical knowledge.**
        - **Avoid including any misleading or incorrect information.**
        - Do not provide the final answer.

        ### Output Format
        Provide your enhanced information in valid JSON format:
        {"enhanced_information": "Your synthesized and enhanced information"}

        """

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert assisting in preparing information for answering medical questions."},
                {"role": "user", "content": prompt}
            ]
            response = self._get_completion(messages)
            self.logger.debug(f"Raw LLM response: {response}")

            # 解析大模型的回复，确保是有效的JSON格式
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]

            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                response = response[start_idx:end_idx]

            result = json.loads(response)

            if "enhanced_information" not in result:
                raise ValueError("Missing 'enhanced_information' in response")

            # 更新问题对象
            question.enhanced_information = result["enhanced_information"]

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}\nResponse: {response}")
            question.enhanced_information = "Error processing enhanced information"

        except Exception as e:
            self.logger.error(f"Error in enhancing information: {str(e)}, Response: {response}")
            question.enhanced_information = "Error processing enhanced information"

    def answer_with_enhanced_information(self, question: MedicalQuestion) -> None:
        """使用增强后的信息生成最终答案"""
        prompt = f"""As a medical expert in the field of {question.topic_name}, please answer the following question using a structured approach.

        ### Question
        {question.question}

        ### Options
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        if question.enhanced_information:
            prompt += f"\n### Enhanced Information\n{question.enhanced_information}\n"

        prompt += """
        ### Information Priority (from highest to lowest)
        1. Basic medical facts and clinical definitions
        2. Statistical evidence (e.g., prevalence, frequency)
        3. Standard clinical protocols and guidelines
        4. Pathophysiological mechanisms
        5. Supporting relationships and pathways

        ### Required Analysis Structure
        1. Core Question Identification
           - What is the fundamental question being asked?
           - What type of information is needed to answer it?
           - Are there any key terms or concepts that need definition?

        2. Basic Fact Verification
           - What is the essential medical fact/definition needed?
           - Is this fact well-established in medical practice?
           - Does this align with current clinical standards?

        3. Information Assessment
           - How does the enhanced information relate to the core fact?
           - Which parts are essential vs supplementary?
           - Are there any contradictions with basic medical knowledge?

        ### Decision Framework
        1. Start with the most basic, established medical fact that answers the question
        2. Only include additional information if it DIRECTLY supports or challenges this fact
        3. Ignore complex relationships unless they fundamentally change the basic answer
        4. When in doubt, prioritize:
           - Clinical standards over theoretical mechanisms
           - Common medical practice over rare exceptions
           - Direct relationships over indirect ones

        ### Critical Guidelines
        - Focus on answering the specific question asked
        - Avoid being distracted by interesting but nonessential information
        - Remember that complex pathways do not override basic medical facts
        - Consider practical clinical significance over theoretical relationships

        ### Output Format
        Provide your response in valid JSON format:
        {
            "final_analysis": "Your concise analysis following the above structure",
            "answer": "Option letter (opa, opb, opc, or opd only)",
            "confidence": Score between 0-100 based on alignment with established medical facts
        }

        ### Key Requirements
        - Keep analysis focused and relevant
        - Base conclusions primarily on established medical knowledge
        - Use enhanced information as support, not primary evidence
        - Avoid speculation or overcomplication
        """

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert making decisions based on enhanced information."},
                {"role": "user", "content": prompt}
            ]
            response = self._get_completion(messages)
            self.logger.debug(f"Raw LLM response: {response}")

            # 确保响应是有效的 JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]

            # 提取 JSON 部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                response = response[start_idx:end_idx]
            else:
                raise ValueError("No JSON object found in the response.")

            # 解析 JSON
            result = json.loads(response)

            # 提取结果
            question.reasoning = result.get("final_analysis", "")
            question.answer = result.get("answer", "").lower()
            question.confidence = float(result.get("confidence", 0.0))

        except Exception as e:
            self.logger.error(f"Error in final answer generation: {str(e)}", exc_info=True)
            question.reasoning = "Error processing final answer"
            question.answer = ""
            question.confidence = 0.0

if __name__ == '__main__':
    var = {
        "question": "Which of the following hormone is/are under inhibitory of hypothalamus?",
        "topic_name": "Pharmacology",
        "options": {
            "opa": "Prolactin",
            "opb": "Only prolactin",
            "opc": "Only growth hormone",
            "opd": "Both prolactin and growth hormone"
        },
        "correct_answer": "opd"
    }

    question = MedicalQuestion(
        question=var.get('question'),
        options=var.get('options'),
        correct_answer=var.get('correct_answer'),
        topic_name=var.get('topic_name')
    )

    llm = LLMProcessor()
    llm.generate_reasoning_chain(question)
    print(f"casual_nodes_refine:{question.casual_paths_nodes_refine}")
    print(f"question.entities_original_pairs:{question.entities_original_pairs}")

    processor = QueryProcessor()
    processor.process_entity_pairs_enhance(question)
    print(f"question.CG_paths:{question.CG_paths}")
    print(f"question.KG_paths:{question.KG_paths}")


    """question.casual_paths = var.get('casual_paths')
    question.KG_paths = var.get('KG_paths')
    question.CG_paths = var.get('CG_paths')
    question.reasoning = var.get('reasoning')

    print(question.enhanced_information)
    llm.enhance_information(question)
    print("Enhanced Information:")
    print(question.enhanced_information)
    llm.answer_with_enhanced_information(question)
    print("Final Analysis:")
    print(question.reasoning)
    print("Answer:", question.answer)
    print("Confidence:", question.confidence)
    # 在单独的步骤中，使用增强后的信息来回答问题
    # 您可以添加一个新的方法，或者使用现有的方法"""
