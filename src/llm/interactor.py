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
        1. **Causal Identification**: Analyze the provided causal pathways (if any) and identify the core causal entities (start and end) that are critical for understanding the relationships. These pathways are preliminary and may include irrelevant or incorrectly directed entities. If no causal pathways are provided, infer causal entities directly from the question and options.
        2. **Reasoning Chain and Additional Entities**: Develop a logical reasoning chain to answer the question. Identify additional entity pairs (start and end) that are not covered in the causal pathways but are crucial for completing the reasoning chain.

        ### Question
        {question.question}

        ### Options
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        # 添加因果路径（如果有）
        if question.casual_paths:
            prompt += "\n### Provided Causal Relationships for Each Option\n"
            for option in ['opa', 'opb', 'opc', 'opd']:
                if question.casual_paths.get(option):
                    prompt += f"\nOption {option} causal paths:\n"
                    for path in question.casual_paths[option]:
                        prompt += f"- {path}\n"
        else:
            prompt += "\nNo causal relationships are provided for this question."

        # 指定任务与输出格式，添加示例和具体要求
        prompt += f"""

### Task 1: Causal Identification
- From each causal pathway, extract the **start** and **end** entities that form the core causal relationship.
- Focus only on relevant causal entities and ignore redundant or irrelevant nodes.
- If no causal pathways are provided, directly infer the most relevant causal entities (start and end) from the question and options.
- **Ensure that the entities are precise medical terms that can be found in standard medical knowledge graphs like UMLS.**
- **Focus on causal relationships that directly lead to the correct answer ('Decrease in preload').**
- **For example, consider 'Nitrates' leading to 'Venodilation' leading to 'Decrease in preload'.**

### Task 2: Reasoning Chain and Additional Entities
- Create a clear and logical reasoning chain to answer the question. Use the causal relationships if applicable.
- Identify additional **start** and **end** entity pairs that are essential for completing the reasoning chain but are not covered in the provided causal pathways (if any).
- Ensure each reasoning step corresponds to at least one entity pair.
- **Again, use precise medical terms aligned with UMLS standards, and focus on mechanisms related to 'Decrease in preload'.**

        ### Output Format (ensure valid JSON):
        {{
            "causal_analysis": {{
                "start": ["Entity1", "Entity2", "Entity3"],
                "end": ["Entity4", "Entity5", "Entity6"]
            }},
            "reasoning_chain": [
                "Step 1: Analyze...",
                "Step 2: Consider...",
                "Step 3: Finally evaluate..."
            ],
            "additional_entity_pairs": {{
                "start": ["EntityA", "EntityB"],
                "end": ["EntityC", "EntityD"],
                "reasoning": [
                    "EntityA connects to EntityC because...",
                    "EntityB connects to EntityD because..."
                ]
            }}
        }}

        ### Key Requirements
        - Ensure 'start' and 'end' arrays are of **equal length** and correspond **one-to-one**, this is very important!!!.
        - All entities must be **full terms** and belong to the **medical domain**. Avoid abbreviations, shorthand, or non-medical terms. this is very important!!!.
        - **Use precise medical terms and align entities with clinical or UMLS standards where possible.**
        - **Provide entities that can be directly used to query standard medical knowledge graphs like UMLS.**
        - If no causal pathways are provided, infer the most relevant causal entities directly from the question context.
        - Reasoning chains must be logical and accurate, even if incomplete.
        - No repeated entity pairs in additional_entity_pairs and additional_entity_pairs
        """

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
            if "reasoning_chain" not in result or "causal_analysis" not in result:
                raise ValueError("Missing required fields in response")

            # 提取因果分析结果
            if result.get("causal_analysis"):
                question.casual_paths_nodes_refine = {
                    'start': result["causal_analysis"].get("start", []),
                    'end': result["causal_analysis"].get("end", [])
                }
                self.logger.debug(f"Extracted causal analysis: {question.casual_paths_nodes_refine}")

            # 提取推理链
            if result.get("reasoning_chain"):
                question.reasoning_chain = "\n".join(result["reasoning_chain"])
                self.logger.debug(f"Extracted reasoning chain: {question.reasoning_chain}")

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
        - Focus on the most relevant mechanisms and information that directly support the correct answer.
        - Trim irrelevant, redundant, or misleading parts.
        - Merge and integrate relevant data from different sources coherently.
        - Ensure that the enhanced information is accurate, relevant, and logically supports the correct answer.
        - Prepare a concise and coherent summary that can be used to directly answer the question.
        - Do NOT provide the final answer to the question at this stage, nor should you make it obvious which option is correct.

        ### Output Format
        Provide your enhanced information in valid JSON format:
        {"enhanced_information": "Your synthesized and enhanced information"}

        ### Key Requirements
        - Be precise and use correct medical terminology.
        - Emphasize the key mechanisms and information that lead to the correct answer.
        - Minimize or exclude information that is not directly related to the correct answer.
        - Ensure that the enhanced information is logically coherent and directly relevant to the question.
        - **Ensure all information is accurate and based on established medical knowledge.**
        - **Avoid including any misleading or incorrect information.**
        - Do not include irrelevant information.
        - Do not provide the final answer.
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
        "question": "Major mechanism of action of nitrates in acute attack of angina is:",
        "topic_name": "Pharmacology",
        "options": {
            "opa": "Coronary vasodilation",
            "opb": "Decrease in preload",
            "opc": "Decrease in afterload",
            "opd": "Decrease in heart rate"
        },
        "correct_answer": "opb",
        "casual_paths": {
            # 您的因果路径数据
        },
        "KG_paths": [
            "(Nitrates)-CAUSES->(Venodilation)-REDUCES->(Preload)",
            "(Nitrates)-CAUSES->(Coronary vasodilation)-INCREASES->(Oxygen supply)"
        ],
        "CG_paths": [
            "(Nitrates)-DECREASES->(Myocardial oxygen demand)-RELIEVES->(Angina)",
            "(Nitrates)-CAUSES->(Venous pooling)-REDUCES->(Preload)"
        ],
        "reasoning": "Nitrates primarily cause vasodilation, which decreases preload and reduces myocardial oxygen demand, relieving angina symptoms."
    }

    question = MedicalQuestion(
        question=var.get('question'),
        options=var.get('options'),
        correct_answer=var.get('correct_answer'),
        topic_name=var.get('topic_name')
    )
    question.casual_paths = var.get('casual_paths')
    question.KG_paths = var.get('KG_paths')
    question.CG_paths = var.get('CG_paths')
    question.reasoning = var.get('reasoning')

    llm = LLMProcessor()
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
    # 您可以添加一个新的方法，或者使用现有的方法
