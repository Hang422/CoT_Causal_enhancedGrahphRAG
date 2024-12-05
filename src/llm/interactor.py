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


def _clean_json_response(response: str) -> str:
    """清理LLM响应中的JSON文本"""
    response = response.strip()
    if response.startswith('```json'):
        response = response[7:]
    if response.endswith('```'):
        response = response[:-3]

    start_idx = response.find('{')
    end_idx = response.rfind('}') + 1
    if start_idx != -1 and end_idx != -1:
        return response[start_idx:end_idx]
    raise ValueError("No JSON object found in response")


class LLMProcessor:
    """处理与LLM的所有交互"""

    def __init__(self):
        """使用全局配置初始化处理器"""
        self.client = OpenAI(api_key=config.openai["api_key"])
        self.model = config.openai.get("model")
        self.temperature = config.openai.get("temperature")
        self.logger = config.get_logger("llm_interaction")
        self.logger.info(f"Initialized LLM interaction with model: {self.model}")
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
            # self.gpt_logger.log_interaction(messages, response_text, "completion")
            return response_text
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}", exc_info=True)
            raise

    def direct_answer(self, question: MedicalQuestion) -> None:
        """直接回答问题，不使用任何额外知识"""
        if question.topic_name is not None:
            prompt = f"""You are a medical expert specializing in the field of {question.topic_name}."""
        else:
            prompt = f"""You are a medical expert."""

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
            "answer": "Option key from the available options (only key,like opa, no additional information)",
            "confidence": Score between 0-100 based on alignment with established medical facts
        }
        """

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert making decisions based on enhanced information."},
                {"role": "user", "content": prompt}
            ]
            response = self._get_completion(messages)
            response = _clean_json_response(response)
            result = json.loads(response)

            question.analysis = result.get("final_analysis", "")
            question.answer = result.get("answer", "").lower()
            question.confidence = float(result.get("confidence", 0.0))

        except Exception as e:
            self.logger.error(f"Error in final answer generation: {str(e)}")
            question.analysis = "Error processing final answer"
            question.answer = ""
            question.confidence = 0.0

    def generate_reasoning_chain(self, question: MedicalQuestion) -> None:
        """基于因果路径生成推理链和需要验证的实体对"""

        if question.topic_name is not None:
            prompt = f"""You are a medical expert specializing in the field of {question.topic_name}."""
        else:
            prompt = f"""You are a medical expert."""

        # 构建初始 Prompt
        prompt += f""" Your task involves two main objectives:

        1. **Generate Causal Chains (Causal Analysis)**: Generate a chain of thought by identifying direct relationships between **medical entities** (e.g., symptoms, organisms, anatomical structures, pathways) relevant to answering the question.

        2. **Infer Additional Entity Pairs**: Identify additional entity relationships that need to be retrieved to further confirm or exclude options, enhancing the certainty of the answer.

        ### Important Note on Provided Causal Relationships
        - The causal relationships provided in `question.casual_paths` are **preliminary and rough**. They may include irrelevant or incorrect information and are only intended as an initial reference.
        - If no causal relationships are provided or if they are unhelpful, generate the causal chain directly from the question and options based on your domain knowledge and expertise.

        ### Question
        {question.question}
        """

        if question.is_multi_choice:
            prompt += "\n### Options\n"
            for key, text in question.options.items():
                prompt += f"{key}. {text}\n"

        # 添加因果路径（如果提供）
        if question.initial_casual_paths is not None and len(question.initial_casual_paths) > 0:
            prompt += "\n### Provided Causal Relationships for Each Option\n"
            for key, paths in question.initial_casual_paths.items():
                if paths:
                    prompt += f"\nOption {key} related causal paths:\n"
                    for path in paths:
                        prompt += f"- {path}\n"
        else:
            prompt += "\nNo causal relationships are provided for this question."

        # 指定任务和输出格式
        prompt += f"""

        ### Task
        Your task is to:
        1. **Causal Analysis**:
           - Generate a chain of thought by identifying `start` and `end` **medical entities** that form direct relationships relevant to answering the question.
           - **Ensure all entities are precise medical nouns without any modifiers, adjectives, abbreviations, or vague descriptors.**

        2. **Additional Entity Pairs**:
           - Identify additional entity relationships (start and end) that need to be retrieved to further confirm or exclude options.
           - These additional entities should help enhance the certainty of the answer or assist in eliminating incorrect options.

        ### Output Format
        The output should be in the following JSON format:

        {{
            "causal_analysis": {{
                "start": ["Entity1", "Entity2", ...],
                "end": ["Entity3", "Entity4", ...]
            }},
            "additional_entity_pairs": {{
                "start": ["EntityA", "EntityB", ...],
                "end": ["EntityC", "EntityD", ...]
            }}
        }}

        ### Key Guidelines
        1. Ensure that the `start` and `end` arrays in both `causal_analysis` and `additional_entity_pairs` are **one-to-one aligned** and represent meaningful relationships. Notice: one to one !!!!
        2. Use **precise medical noun entities** for all terms, avoiding any modifiers, adjectives, abbreviations, or vague descriptors.
        3. The `causal_analysis` should reflect your chain of thought leading to the answer.
        4. The `additional_entity_pairs` should include entities that, if retrieved, would further confirm the correct option or help exclude the incorrect ones.
        5. **Do not include any explanations or justifications** in the output; only provide the structured `start` and `end` pairs.
        6. Ensure all terms can be directly queried in UMLS or similar medical knowledge graphs.

        ### Example

        **Question**: Which of the following hormone is/are under inhibitory control of the hypothalamus?

        **Options**:
        A. Prolactin  
        B. Only prolactin  
        C. Only growth hormone  
        D. Both prolactin and growth hormone

        **Task Execution**:

        1. **Causal Analysis** (Chain of Thought):
           - **Start**: ["Hypothalamus", "Hypothalamus"]
           - **End**: ["Prolactin Inhibitory Hormone", "Growth Hormone Inhibitory Hormone"]

        2. **Additional Entity Pairs** (To Confirm or Exclude Options):
           - **Start**: ["Prolactin"]
           - **End**: ["Anterior Pituitary Gland"]
        ### Output Format, **Notice: the direction of, from start to end.**
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
        """

        self.logger.debug(f"Prompt sent to LLM:\n{prompt}")

        # 调用 LLM 获取结果的代码保持不变

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert analyzing questions. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            response = self._get_completion(messages)

            # 处理JSON响应
            response = _clean_json_response(response)
            result = json.loads(response)

            if "causal_analysis" in result:
                if len(result["causal_analysis"].get("start", [])) != len(result["causal_analysis"].get("end", [])):
                    self.logger.error(f"Question {hashlib.md5(question.question.encode()).hexdigest()}'s causal analysis did not match one to one format.")
                question.causal_graph.entities_pairs = {
                    'start': result["causal_analysis"].get("start", []),
                    'end': result["causal_analysis"].get("end", [])
                }

            if "additional_entity_pairs" in result:
                if len(result["additional_entity_pairs"].get("start", [])) != len(result["additional_entity_pairs"].get("end", [])):
                    self.logger.error(f"Question {hashlib.md5(question.question.encode()).hexdigest()}'s knowledge analysis did not match one to one format.")
                question.knowledge_graph.entities_pairs = {
                    'start': result["additional_entity_pairs"].get("start", []),
                    'end': result["additional_entity_pairs"].get("end", [])
                }

        except Exception as e:
            self.logger.error(f"Error in reasoning chain generation: {str(e)}")
            question.casual_paths_nodes_refine = {'start': [], 'end': []}
            question.entities_original_pairs = {'start': [], 'end': []}

    def enhance_information(self, question: MedicalQuestion) -> None:
        """融合所有信息，生成增强后的信息"""
        self.logger.debug(f"Starting reasoning chain generation")

        if question.topic_name is not None:
            prompt = f"""You are a medical expert specializing in {question.topic_name}."""
        else:
            prompt = """You are a medical expert."""

        prompt += f"""Your task is to analyze this medical question and generate entity relationship pairs that, when verified and queried in medical knowledge bases, would provide strong evidence for answering the question. These entity pairs will be used to retrieve concrete relationships from knowledge graphs to support the answer.

        ### Question
        {question.question}

        ### Options
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        prompt += """
        ### Task
        1. First, determine if answering the question requires:
           - Verifying causal relationships (e.g., "What causes X?", "What effect does Y have?", "How does Z work?")
           - Verifying non-causal relationships (e.g., location, composition, function, classification)
           - Both types of relationships

        2. Generate entity pairs that, when verified in knowledge bases, would provide evidence to answer the question:
           - For causal relationships: Use "causal_analysis" with start->end entity pairs
             These pairs will be verified in causal knowledge graphs to find cause-effect evidence
           - For non-causal relationships: Use "additional_entity_pairs" with start->end entity pairs
             These pairs will be verified in general medical knowledge graphs to find supporting evidence
           - Each entity pair represents a relationship that, when verified, helps answer the question
           - All entities must be standard medical terms (those with UMLS CUIs)

        3. The entity pairs you generate will be used to:
           - Query medical knowledge bases for relationship evidence
           - Validate the existence and strength of relationships
           - Build evidence chains supporting or ruling out different options
           - Provide concrete, verifiable paths to the correct answer

        ### Output Format
        {
            "causal_analysis": {
                "start": ["Entity1", "Entity2"],  # Entities to verify causal relationships
                "end": ["Entity3", "Entity4"]     # Each maps one-to-one with start entities
            },
            "additional_entity_pairs": {
                "start": ["EntityA", "EntityB"],  # Entities to verify non-causal relationships
                "end": ["EntityC", "EntityD"]     # Each maps one-to-one with start entities
            }
        }

        ### Key Requirements
        - Use only concrete medical entities that can be found in knowledge bases
        - **Maintain one-to-one mapping between start and end entities**
        - Each entity pair should, when verified, provide clear evidence for answering the question
        - Generated pairs must be verifiable through knowledge graph queries
        - All pairs should contribute to distinguishing between the given options
        """

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert assisting in preparing information for answering medical questions."},
                {"role": "user", "content": prompt}
            ]
            response = self._get_completion(messages)

            # 处理JSON响应
            response = _clean_json_response(response)
            result = json.loads(response)
            question.enhanced_information = result.get("enhanced_information", "")
        except Exception as e:
            self.logger.error(f"Error in information enhancement: {str(e)}")
            question.enhanced_information = "Error processing enhanced information"

    def answer_with_enhanced_information(self, question: MedicalQuestion) -> None:
        """使用增强后的信息生成最终答案"""
        if question.topic_name is not None:
            prompt = f"""You are a medical expert specializing in the field of {question.topic_name}."""
        else:
            prompt = f"""You are a medical expert."""

        prompt = f"""Please answer the following question using a structured approach.

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
            response = _clean_json_response(response)
            result = json.loads(response)

            question.analysis = result.get("final_analysis", "")
            question.answer = result.get("answer", "").lower()
            question.confidence = float(result.get("confidence", 0.0))

        except Exception as e:
            self.logger.error(f"Error in final answer generation: {str(e)}")
            question.analysis = "Error processing final answer"
            question.answer = ""
            question.confidence = 0.0


if __name__ == '__main__':
    var = {
        "question": "Are group 2 innate lymphoid cells ( ILC2s ) increased in chronic rhinosinusitis with nasal polyps or eosinophilia?",
        "context": """{ "contexts": [ "Chronic rhinosinusitis (CRS) is a heterogeneous disease with an uncertain pathogenesis. Group 2 innate lymphoid cells (ILC2s) represent a recently discovered cell population which has been implicated in driving Th2 inflammation in CRS; however, their relationship with clinical disease characteristics has yet to be investigated.", "The aim of this study was to identify ILC2s in sinus mucosa in patients with CRS and controls and compare ILC2s across characteristics of disease.", "A cross-sectional study of patients with CRS undergoing endoscopic sinus surgery was conducted. Sinus mucosal biopsies were obtained during surgery and control tissue from patients undergoing pituitary tumour resection through transphenoidal approach. ILC2s were identified as CD45(+) Lin(-) CD127(+) CD4(-) CD8(-) CRTH2(CD294)(+) CD161(+) cells in single cell suspensions through flow cytometry. ILC2 frequencies, measured as a percentage of CD45(+) cells, were compared across CRS phenotype, endotype, inflammatory CRS subtype and other disease characteristics including blood eosinophils, serum IgE, asthma status and nasal symptom score.", "35 patients (40% female, age 48 ± 17 years) including 13 with eosinophilic CRS (eCRS), 13 with non-eCRS and 9 controls were recruited. ILC2 frequencies were associated with the presence of nasal polyps (P = 0.002) as well as high tissue eosinophilia (P = 0.004) and eosinophil-dominant CRS (P = 0.001) (Mann-Whitney U). They were also associated with increased blood eosinophilia (P = 0.005). There were no significant associations found between ILC2s and serum total IgE and allergic disease. In the CRS with nasal polyps (CRSwNP) population, ILC2s were increased in patients with co-existing asthma (P = 0.03). ILC2s were also correlated with worsening nasal symptom score in CRS (P = 0.04)." ], "labels": [ "BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS" ], "meshes": [ "Adult", "Aged", "Antigens, Surface", "Case-Control Studies", "Chronic Disease", "Eosinophilia", "Female", "Humans", "Hypersensitivity", "Immunity, Innate", "Immunoglobulin E", "Immunophenotyping", "Leukocyte Count", "Lymphocyte Subsets", "Male", "Middle Aged", "Nasal Mucosa", "Nasal Polyps", "Neutrophil Infiltration", "Patient Outcome Assessment", "Rhinitis", "Sinusitis", "Young Adult" ] }""",
        "correct_answer": "yes",
    }

    var1 = {
        "causal_analysis": {
            "start": [
                "Group 2 innate lymphoid cells",
                "Group 2 innate lymphoid cells",
                "Group 2 innate lymphoid cells",
                "Eosinophilia",
                "Chronic rhinosinusitis with nasal polyps"
            ],
            "end": [
                "Th2 inflammation",
                "Increased eosinophils in sinus mucosa",
                "Worsening nasal symptom score",
                "Chronic rhinosinusitis",
                "Increased Group 2 innate lymphoid cells"
            ]
        },
        "additional_entity_pairs": {
            "start": [
                "Th2 inflammation",
                "Nasal polyps",
                "Asthma",
                "Blood eosinophilia"
            ],
            "end": [
                "Increased Group 2 innate lymphoid cells",
                "Worsening nasal symptom score",
                "Increased Group 2 innate lymphoid cells",
                "Eosinophilic chronic rhinosinusitis"
            ]
        }
    }
    question = MedicalQuestion(
        question='Which enzyme is responsible for converting angiotensin I to angiotensin II?',
        options={
            "opa": "Renin",
            "opb": "Angiotensin Converting Enzyme",
            "opc": "Chymase",
            "opd": "Carboxypeptidase"},
        correct_answer=var.get('correct_answer'),
        is_multi_choice=False
    )

    llm = LLMProcessor()
    llm.generate_reasoning_chain(question)

    processor = QueryProcessor()
    processor.process_all_entity_pairs_enhance(question)
    print(f"question.causal_graph.entities_pairs: {question.causal_graph.entities_pairs}")
    print(f"question.knowledge_graph.entities_pairs: {question.knowledge_graph.entities_pairs}")
    print(f"question.CG_paths:{question.causal_graph.paths}")
    print(f"question.KG_paths:{question.knowledge_graph.paths}")

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
