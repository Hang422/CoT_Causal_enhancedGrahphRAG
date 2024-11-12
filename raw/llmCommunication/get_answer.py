import json
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


def _construct_prompt_with_casual_paths(question: str,
                                        options: Dict[str, str],
                                        causal_paths: List[str]) -> str:
    """构建包含因果路径的提示"""
    prompt = f"""As a medical expert, please help answer this multiple choice question based on the following information:

Question: {question}

Options:
"""
    for opt, text in options.items():
        prompt += f"{opt}. {text}\n"

    prompt += "\nRelevant causal relationships retrieved from the medical knowledge graph:\n"
    for path in causal_paths:
        prompt += f"- {path}\n"

    prompt += """
Based on the question, options, and causal relationships provided above, please analyze and select the most appropriate answer. Respond in the following format:

Analysis: (Detailed explanation of your reasoning process, including how you utilized the causal relationships to reach your conclusion)
Answer: (Option letter)
Confidence: (A number from 0-100 indicating your confidence in this answer)"""

    return prompt


def _construct_prompt_without_paths(question: str,
                                    options: Dict[str, str]) -> str:
    """构建不包含因果路径的提示"""
    prompt = f"""As a medical expert, please help answer this multiple choice question:

Question: {question}

Options:
"""
    for opt, text in options.items():
        prompt += f"{opt}. {text}\n"

    prompt += """
Please analyze and select the most appropriate answer. Respond in the following format:

Analysis: (Detailed explanation of your reasoning process)
Answer: (Option letter)
Confidence: (A number from 0-100 indicating your confidence in this answer)"""

    return prompt




class MedicalQAGPT:
    """医学问答GPT接口类"""

    def __init__(self, api_key: str):
        """初始化GPT接口"""
        self.client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_gpt_response(self,
                          prompt: str,
                          temperature: float = 0.3) -> Dict:
        """获取GPT回答"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a medical expert helping to answer multiple choice questions based on medical knowledge."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )

            response_text = response.choices[0].message.content

            # 解析响应
            analysis = ""
            answer = ""
            confidence = 0

            for line in response_text.split('\n'):
                if line.startswith('Analysis:'):
                    analysis = line.replace('Analysis:', '').strip()
                elif line.startswith('Answer:'):
                    answer = line.replace('Answer:', '').strip().lower()
                elif line.startswith('Confidence:'):
                    try:
                        confidence = int(line.replace('Confidence:', '').strip())
                    except ValueError:
                        confidence = 0

            return {
                'analysis': analysis,
                'answer': answer,
                'confidence': confidence,
                'full_response': response_text
            }

        except Exception as e:
            print(f"调用GPT API时出错: {str(e)}")
            return {
                'analysis': "分析过程中出现错误",
                'answer': None,
                'confidence': 0,
                'error': str(e)
            }

    def get_answer_with_paths(self,
                              question: str,
                              options: Dict[str, str],
                              causal_paths: List[str],
                              temperature: float = 0.3) -> Dict:
        """使用因果路径获取答案"""
        prompt = _construct_prompt_with_casual_paths(question, options, causal_paths)
        return self._get_gpt_response(prompt, temperature)

    def get_answer_without_paths(self,
                                 question: str,
                                 options: Dict[str, str],
                                 temperature: float = 0.3) -> Dict:
        """不使用因果路径获取答案"""
        prompt = _construct_prompt_without_paths(question, options)
        return self._get_gpt_response(prompt, temperature)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_reasoning_and_pairs(self,
                                question: str,
                                options: Dict[str, str],
                                causal_paths: List[str]) -> Dict:
        """
        基于问题和已知因果关系生成推理链和需要查询的实体对

        Args:
            question: 问题文本
            options: 选项字典，格式为 {'a': '选项内容', 'b': '选项内容', ...}
            causal_paths: 已知的因果路径列表（100%正确的因果关系）

        Returns:
            Dict: 包含推理链和需要查询的实体对的字典
                 格式：{
                     'reasoning_chain': List[str],  # 推理步骤列表
                     'entity_pairs': List[Dict[str, str]]  # 实体对列表
                 }
        """
        prompt = f"""Based on the medical question and given causal relationships below, first develop a reasoning chain to answer the question, then identify what additional entity relationships we need to verify.

        Question: {question}

        Options:
        """
        for opt, text in options.items():
            prompt += f"{opt}. {text}\n"

        prompt += "\nAvailable causal relationships (100% accurate, but may not all be relevant):\n"
        for path in causal_paths:
            prompt += f"- {path}\n"

        prompt += """
        Return your response in the following JSON format ONLY:
        {
            "reasoning_chain": [
                "step1: First key point in answering this question...",
                "step2: How this point leads to evaluating specific options...",
                "step3: What additional information would strengthen or confirm our answer..."
            ],
            "entity_pairs": [
                {
                    "start": "entity1",
                    "end": "entity2",
                    "reasoning": "how verifying this relationship would help confirm our answer"
                }
            ]
        }

        Guidelines:
        - Reasoning chain should focus on how to answer the question
        - You don't need to use all given causal relationships - only use those that are actually relevant
        - Each reasoning step should be clear and directly contribute to answering the question
        - Entity pairs should be pairs we need to verify to be more confident in our answer
        - Use specific medical terms from the question and options
        - Include 2-4 most critical entity pairs to check
        - Return only the JSON format, no additional explanations
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a medical expert analyzing questions using verified causal relationships."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            response_text = response.choices[0].message.content

            # 解析JSON响应
            import json
            try:
                # 找到第一个 { 和最后一个 } 之间的内容
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    result = json.loads(json_str)
                    return {
                        'reasoning_chain': result.get('reasoning_chain', []),
                        'entity_pairs': result.get('entity_pairs', [])
                    }
                else:
                    print("No valid JSON found in response")
                    return {'reasoning_chain': [], 'entity_pairs': []}
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}")
                return {'reasoning_chain': [], 'entity_pairs': []}

        except Exception as e:
            print(f"调用GPT API时出错: {str(e)}")
            return {'reasoning_chain': [], 'entity_pairs': []}

    def get_answer_with_reasoning_paths(self,
                                        question: str,
                                        options: Dict[str, str],
                                        reasoning_chain: List[str],
                                        entity_pairs_paths: List[Dict]) -> Dict:
        """
        基于问题、推理链和实体路径获取答案

        Args:
            question: 问题文本
            options: 选项字典
            reasoning_chain: 推理步骤列表
            entity_pairs_paths: 实体对及其路径列表

        Returns:
            Dict: 包含分析、答案和置信度的字典
        """
        prompt = f"""As a medical expert, please analyze this question using the provided reasoning chain and causal paths:

    Question: {question}

    Options:
    """
        for opt, text in options.items():
            prompt += f"{opt}. {text}\n"

        prompt += "\nPrevious reasoning chain:\n"
        for step in reasoning_chain:
            prompt += f"- {step}\n"

        prompt += "\nVerified causal paths between key entities:\n"
        for pair in entity_pairs_paths:
            start_entity = pair['start']['text']
            end_entity = pair['end']['text']
            prompt += f"\nRelationship between '{start_entity}' and '{end_entity}':\n"
            for path in pair['paths']:
                prompt += f"- {path}\n"

        prompt += """
    Based on all this information, please:
    1. Analyze how the verified causal paths support or contradict the reasoning chain
    2. Determine the answer with this enriched understanding

    Respond in the following format:

    Final Analysis: (Explain how the causal paths inform your decision, specifically addressing how they support or modify the initial reasoning)
    Answer: (Option letter)
    Confidence: (A number from 0-100, considering both the reasoning and causal evidence)
    Explanation: (Why this confidence level was chosen)
    """

        return self._get_gpt_response(prompt, temperature=0.3)

def main():
    """示例用法"""
    api_key = "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"  # 替换成你的实际API密钥

    qa_gpt = MedicalQAGPT(api_key=api_key)

    question = "Which of the following is not true for myelinated nerve fibers:"
    options = {
        'a': "Impulse through myelinated fibers is slower than non-myelinated fibers",
        'b': "Membrane currents are generated at nodes of Ranvier",
        'c': "Saltatory conduction of impulses is seen",
        'd': "Local anesthesia is effective only when the nerve is not covered by myelin sheath"
    }
    causal_paths = [
        "Myelinated nerve fiber-[AFFECTS]-Aging-[AFFECTS]-collagen",
        "Myelinated nerve fiber-[AFFECTS]-Signaling Molecule-[AFFECTS]-Anatomic Node",
        "Myelinated nerve fiber-[AFFECTS]-calcium-[AUGMENTS]-Biological Evolution-[AFFECTS]-Orthoptera",
        "Myelinated nerve fiber-[AFFECTS]-Aging-[AFFECTS]-Myelin Sheath"
    ]

    # 使用因果路径
    llm = MedicalQAGPT("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" )
    required_pairs = llm.get_reasoning_and_pairs(question, options, causal_paths)
    print(required_pairs['entity_pairs'])

    """result1 = qa_gpt.get_answer_with_paths(question, options, causal_paths)
    print("\n使用因果路径的结果:")
    print(f"分析: {result1['analysis']}")
    print(f"答案: {result1['answer']}")
    print(f"置信度: {result1['confidence']}")

    # 不使用因果路径
    result2 = qa_gpt.get_answer_without_paths(question, options)
    print("\n不使用因果路径的结果:")
    print(f"分析: {result2['analysis']}")
    print(f"答案: {result2['answer']}")
    print(f"置信度: {result2['confidence']}")"""


if __name__ == "__main__":
    api_key = "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"
    llm = MedicalQAGPT(api_key)

    # 读取问题数据
    questions_df = pd.read_csv('../../data/testdata/samples.csv')

    # 读取因果路径数据
    with open('../../data/results/middleResults/causal_paths_v0.json', 'r', encoding='utf-8') as f:
        causal_paths_data = json.load(f)

    # 存储所有结果
    results = []

    # 处理每个问题
    for index, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing questions"):
        try:
            # 提取问题和选项
            question = row['question']
            options = {
                'a': row['opa'],
                'b': row['opb'],
                'c': row['opc'],
                'd': row['opd']
            }

            # 获取该问题的所有因果路径
            all_paths = []
            if index < len(causal_paths_data):
                paths_data = causal_paths_data[index]['paths']
                for option in ['option_a', 'option_b', 'option_c', 'option_d']:
                    all_paths.extend(paths_data[option])

            # 使用已有函数获取思维链和实体对
            analysis_result = llm.get_reasoning_and_pairs(question, options, all_paths)

            # 保存结果
            result = {
                'question_index': index,
                'question': question,
                'options': options,
                'causal_paths': all_paths,
                'analysis_result': analysis_result
            }
            results.append(result)

        except Exception as e:
            print(f"\nError processing question {index}: {str(e)}")
            continue

    # 保存结果到文件
    output_path = '../temp/pairs.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}")