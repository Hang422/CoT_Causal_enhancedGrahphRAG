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
            self.gpt_logger.log_interaction(messages, response_text, "completion")
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

    def causal_only_answer(self, question: MedicalQuestion) -> None:
        """直接基于因果图回答问题"""
        if question.topic_name is not None:
            prompt = f"""You are a medical expert specializing in the field of {question.topic_name}."""
        else:
            prompt = f"""You are a medical expert."""

        prompt += f"""Your task is to analyze and answer this {'' if not question.is_multi_choice else 'multiple choice'} question with the help of the causal graph information provided.

    ### Question
    {question.question}
    """
        if question.is_multi_choice:
            prompt += "\n### Options:\n"
            for key, text in question.options.items():
                prompt += f"{key}. {text}\n"

        if question.initial_causal_graph and question.initial_causal_graph.paths:
            prompt += "\n### Causal Graph Paths\n"
            for path in question.initial_causal_graph.paths:
                prompt += f"- {path}\n"
        else:
            prompt += "\n### Causal Graph Paths\nNo causal paths provided.\n"

        prompt += """
    ### Output Instructions
    - Analyze the question based on the causal graph paths provided.
    - Your analysis should be concise and explain how the causal paths support the chosen answer.

    ### Output Format
    Provide your response in valid JSON format:
    {   
        "final_analysis": "Your concise analysis explaining the reasoning based on the causal paths",
        "answer": "Option key from the available options (e.g., opa, opb)",
        "confidence": A score between 0-100 based on how strongly the causal paths support your answer
    }
    """

        try:
            messages = [
                {"role": "system",
                 "content": "You are a medical expert making decisions solely based on causal graph paths."},
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
            prompt = f"You are a medical expert specializing in the field of {question.topic_name}."
        else:
            prompt = f"You are a medical expert."

        prompt += f"""
        Your task involves two main objectives:

        1. **Generate Causal Chains (Causal Analysis)**: 
           - Identify direct relationships between **medical entities** that are specifically relevant to answering the question.
           - Prioritize standard medical knowledge and clinical consensus. The provided causal paths are only a secondary reference.
           - If initial causal graph relationships do not help in distinguishing between the given options or conflict with well-established medical facts, ignore them.

        2. **Infer Additional Entity Pairs**: 
           - Identify additional entity pairs that, if verified, would help confirm the correct option or exclude incorrect options.
           - These entity pairs must have direct clinical relevance to the question, taking into account the patient's characteristics (e.g., symptoms, risk factors, demographics) and the conditions described by the options.
           - Only include pairs that are known from standard medical knowledge to be involved in differentiating between the conditions presented in the options.

        ### Important Note on Provided Causal Relationships
        - The causal relationships in `question.casual_paths` are rough and may be misleading.
        - Always rely on standard medical knowledge as the primary source of truth.
        - If the provided paths are not directly useful for making a differential diagnosis among the given options, do not use them.

        ### Question
        {question.question}
        """

        if question.is_multi_choice:
            prompt += "\n### Options\n"
            for key, text in question.options.items():
                prompt += f"{key}. {text}\n"

        if question.initial_causal_graph.paths != "There is no obvious causal relationship.":
            prompt += "\n### Initial Causal Graph Paths (May be irrelevant or misleading)\n"
            for path in question.initial_causal_graph.paths:
                prompt += f"- {path}\n"
        else:
            prompt += "\nNo causal relationships are provided for this question."

        prompt += f"""
        ### Task
        Your task is to:
        1. **Causal Analysis**:
           - Start from the patient's presentation and the conditions listed in the options.
           - Identify `start` and `end` entities that form medically relevant relationships. For example, consider key symptoms, risk factors, and pathophysiological links that help differentiate one option from another.
           - Use **precise medical nouns** only (no modifiers, abbreviations, vague terms).
           - Ignore any provided causal info that does not align with standard medical knowledge or does not help in differentiating the options.

        2. **Additional Entity Pairs**:
           - Include only those entity pairs that, if retrieved or verified, would provide further clarity in confirming the correct choice or excluding incorrect ones.
           - These pairs must be directly related to known differentiating factors among the conditions listed in the options.

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
        1. Ensure one-to-one alignment in both `causal_analysis` and `additional_entity_pairs`.
        2. All entities must be precise medical nouns, suitable for queries in medical knowledge bases (e.g., UMLS).
        3. The `causal_analysis` must reflect standard medical reasoning that is relevant and can help distinguish between the given options.
        4. The `additional_entity_pairs` must be clinically relevant and aimed at confirming or excluding specific options.
        5. Do not provide explanations or justifications, only the JSON structure.
        6. If initial causal info is misleading or not helpful, disregard it.

        """
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
                    self.logger.info(
                        f"Question {hashlib.md5(question.question.encode()).hexdigest()}'s causal analysis did not match one to one format.")
                question.causal_graph.entities_pairs = {
                    'start': result["causal_analysis"].get("start", []),
                    'end': result["causal_analysis"].get("end", [])
                }

            if "additional_entity_pairs" in result:
                if len(result["additional_entity_pairs"].get("start", [])) != len(
                        result["additional_entity_pairs"].get("end", [])):
                    self.logger.info(
                        f"Question {hashlib.md5(question.question.encode()).hexdigest()}'s knowledge analysis did not match one to one format.")
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
        self.logger.debug(f"Starting information enhancement")

        if question.topic_name is not None:
            prompt = f"You are a medical expert specializing in the field of {question.topic_name}."
        else:
            prompt = f"You are a medical expert."

        prompt += f"""Your task is to analyze and enhance medical information with extreme precision and attention to detail. You must ensure that the enhanced information maintains complete medical accuracy while supporting the reasoning process.

        ### Question
        {question.question}

        ### Options
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        # 添加路径信息
        if question.initial_causal_graph.paths != "There is no obvious causal relationship.":
            prompt += "\n### Initial Causal Graph Paths\n"
            for path in question.initial_causal_graph.paths:
                prompt += f"- {path}\n"

        if question.causal_graph.paths != "There is no obvious causal relationship.":
            prompt += "\n### Additional Causal Graph Paths\n"
            for path in question.causal_graph.paths:
                prompt += f"- {path}\n"

        if question.knowledge_graph.paths:
            prompt += "\n### Knowledge Graph Paths\n"
            for path in question.knowledge_graph.paths:
                prompt += f"- {path}\n"

        prompt += """
        ### Critical Requirements
        1. **Maintain Complete Medical Accuracy**:
           - Every statement must be verifiable against standard medical knowledge
           - Do not oversimplify complex medical relationships
           - Preserve all critical diagnostic criteria and pathognomonic features
           - Never make assumptions or generalizations that could lead to diagnostic errors

        2. **Comprehensive Information Integration**:
           - Include ALL relevant pathophysiological mechanisms
           - Preserve specific distinguishing features from provided paths
           - Maintain the full context of medical relationships
           - Never exclude crucial diagnostic or clinical details

        3. **Explicit Relationship Validation**:
           - Verify each causal relationship against core medical principles
           - Ensure all stated mechanisms are scientifically accurate
           - Preserve the complexity of multi-factor relationships
           - Flag and exclude any contradictory or inconsistent relationships

        4. **Clinical Context Preservation**:
           - Maintain all clinically relevant temporal relationships
           - Preserve diagnostic hierarchy and differential considerations
           - Keep all pertinent negative findings
           - Include relevant conditioning factors and exceptions

        ### Quality Control Checklist
        Before finalizing your response, verify that the enhanced information:
        1. [ ] Contains no oversimplified medical concepts
        2. [ ] Preserves all critical diagnostic criteria
        3. [ ] Maintains accurate physiological mechanisms
        4. [ ] Includes all relevant differential considerations
        5. [ ] Preserves complex medical relationships
        6. [ ] Avoids inappropriate generalizations
        7. [ ] Maintains clinical context
        8. [ ] Includes all necessary qualifying conditions

        ### Output Format
        {
            "enhanced_information": "Your comprehensive medical synthesis supporting accurate clinical reasoning"
        }

        ### Important Notes:
        - If you're unsure about any relationship or mechanism, preserve the original complexity rather than simplifying
        - Never sacrifice medical accuracy for clarity
        - Maintain all clinically relevant caveats and exceptions
        - Do not exclude information just because it seems complex
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
            question.enhanced_information = result.get("enhanced_information", "")

        except Exception as e:
            self.logger.error(f"Error in enhancement generation: {str(e)}")

    def answer_with_enhanced_information(self, question: MedicalQuestion) -> None:
        """使用增强后的信息生成最终答案"""
        if question.topic_name is not None:
            prompt = f"You are a medical expert specializing in the field of {question.topic_name}."
        else:
            prompt = f"You are a medical expert."

        prompt += f"""Your task is to accurately answer the question by prioritizing established medical knowledge and using the enhanced information as secondary support.

        ### Question
        {question.question}

        ### Options
        """
        for key, text in question.options.items():
            prompt += f"{key}. {text}\n"

        if question.enhanced_information:
            prompt += f"\n### Enhanced Information\n{question.enhanced_information}\n"

        prompt += """
        ### Information Usage Guidelines
        1. **Priority of Information**:
           - Established medical facts are primary and overriding.
           - Enhanced information is secondary; use it only if it does not conflict with standard medical knowledge.
           - If enhanced information contradicts known medical facts, ignore the conflicting part.

        2. **Analysis Structure**:
           1. Identify the core medical question.
           2. Verify basic medical facts from standard medical knowledge.
           3. Use relevant parts of enhanced information to support your conclusion if they align with medical consensus.
           4. Discard any misleading or irrelevant details.

        3. **Decision Framework**:
           - Base the final answer on the most straightforward, medically accepted fact.
           - Enhanced information is only a supplement. 
           - Focus on what a clinically practicing physician would consider the correct answer.

        ### Output Format
        {
          "final_analysis": "A concise analysis stating the medical reasoning behind the chosen answer",
          "answer": "Option letter (opa, opb, opc, or opd)",
          "confidence": A score between 0-100
        }

        ### Key Guidelines
        - Do not overcomplicate.
        - Remain consistent with clinical and medical standards.
        - Use enhanced information only if it complements established facts."""

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
