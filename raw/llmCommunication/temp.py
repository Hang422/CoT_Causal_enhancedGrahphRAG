import json
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from get_answer import MedicalQAGPT

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
        start_entity = pair['start']['question']
        end_entity = pair['end']['question']
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


# 使用示例:
def process_question_with_paths(question_data: Dict, qa_gpt: MedicalQAGPT) -> Dict:
    """
    处理单个问题的完整流程

    Args:
        question_data: 包含问题、选项、推理链和路径的字典
        qa_gpt: MedicalQAGPT实例

    Returns:
        Dict: 处理结果
    """
    result = qa_gpt.get_answer_with_reasoning_paths(
        question=question_data['question'],
        options=question_data['options'],
        reasoning_chain=question_data['reasoning_chain'],
        entity_pairs_paths=question_data['entity_pairs_paths']
    )

    return {
        'question': question_data['question'],
        'options': question_data['options'],
        'reasoning_chain': question_data['reasoning_chain'],
        'entity_pairs_paths': question_data['entity_pairs_paths'],
        'final_analysis': result['analysis'],
        'answer': result['answer'],
        'confidence': result['confidence'],
        'full_response': result.get('full_response', '')
    }


# Main 函数示例：
def main():
    api_key = "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"  # 替换成你的实际API密钥

    qa_gpt = MedicalQAGPT(api_key)

    # 读取JSON文件
    with open('questions_with_paths.json', 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    # 处理所有问题
    results = []
    for question_data in tqdm(questions_data, desc="处理问题"):
        try:
            result = process_question_with_paths(question_data, qa_gpt)
            results.append(result)
        except Exception as e:
            print(f"处理问题时出错: {str(e)}")
            continue

    # 保存结果
    with open('final_answers.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)