import pandas as pd
from typing import Dict, List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import json
from raw.graphrag.query_processor import EntityPathFinder


class CausalPathQAEvaluator:
    def __init__(self, openai_api_key: str, neo4j_params: Dict[str, str]):
        """初始化评估器

        Args:
            openai_api_key: OpenAI API密钥
            neo4j_params: Neo4j连接参数
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.path_finder = EntityPathFinder(neo4j_params)

    def load_data(self, csv_path: str, cui_path: str) -> tuple:
        """加载问题数据和CUI数据

        Args:
            csv_path: CSV文件路径
            cui_path: CUI JSON文件路径

        Returns:
            tuple: (DataFrame, CUI数据)
        """
        df = pd.read_csv(csv_path)
        with open(cui_path, 'r', encoding='utf-8') as f:
            cui_data = json.load(f)
        return df, cui_data

    def format_question_data(self, row) -> tuple:
        """格式化问题数据"""
        question = row['question']
        options = {
            'a': row['opa'],
            'b': row['opb'],
            'c': row['opc'],
            'd': row['opd']
        }
        correct_answer = chr(97 + int(row['cop']))
        return question, options, correct_answer

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_answer_with_paths(self, question: str, options: Dict[str, str],
                              question_cuis: List[str], option_cuis: Dict[str, List[str]]) -> Dict:
        """Use causal paths to get answer"""
        try:
            # Get causal paths for each option
            paths_by_option = {}
            for opt, cuis in option_cuis.items():
                paths = self.path_finder.find_paths(question_cuis, cuis)
                if paths:
                    paths_by_option[opt] = paths

            # Build prompt
            prompt = (
                f"As a medical expert, please analyze the following question:\n\n"
                f"Question: {question}\n\n"
                "Options:\n"
            )
            for opt, text in options.items():
                prompt += f"{opt}) {text}\n"

            prompt += "\nRelevant causal relationship paths:\n"
            for opt, paths in paths_by_option.items():
                if paths:
                    prompt += f"\nPaths for option {opt.upper()}:\n"
                    for path in paths:
                        prompt += f"- {path}\n"

            prompt += """
    Based on the question, options, and causal relationship paths(100% truth, you should use them to 
    improve and check your answer) provided above:

    1. What is your answer? Please choose one option (a/b/c/d).

    Please format your response exactly as follows:
    Answer: [option letter]
    Confidence: [number]%
    Explanation: [brief reason]
    """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            text = response.choices[0].message.content

            # Parse response
            answer = None
            confidence = 0
            analysis = ""

            for line in text.split('\n'):
                if line.startswith('Answer:'):
                    answer = line.split(':')[1].strip().lower()
                elif line.startswith('Confidence:'):
                    confidence = int(''.join(filter(str.isdigit, line)))
                elif line.startswith('Explanation:'):
                    analysis = line.split(':')[1].strip()

            return {
                'answer': answer,
                'confidence': confidence,
                'analysis': analysis,
                'causal_paths': paths_by_option
            }

        except Exception as e:
            print(f"Error in get_answer_with_paths: {str(e)}")
            return {
                'answer': None,
                'confidence': 0,
                'analysis': str(e),
                'causal_paths': {}
            }
    def evaluate_questions(self, df: pd.DataFrame, cui_data: List[Dict]) -> List[Dict]:
        """评估所有问题"""
        results = []

        for index, (row, cui_row) in enumerate(zip(df.iterrows(), cui_data)):
            row = row[1]  # 获取Series数据
            question, options, correct_answer = self.format_question_data(row)

            print(f"\n处理问题 {index + 1}/{len(df)}:")
            print(f"问题: {question}")

            # 准备CUI数据
            question_cuis = cui_row['question_cuis']
            option_cuis = cui_row['individual_option_cuis']

            # 获取使用因果路径的答案
            result = self.get_answer_with_paths(question, options, question_cuis, option_cuis)

            # 构建结果
            question_result = {
                'question_number': index + 1,
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
                'gpt_raw_answer': result['answer'],
                'gpt_answer': result['answer'],
                'is_correct': result['answer'] == correct_answer if result['answer'] else False,
                'confidence': result['confidence'],
                'analysis': result['analysis'],
                'causal_paths': result['causal_paths']
            }

            results.append(question_result)

            # 打印当前结果
            print(f"GPT答案: {result['answer']}")
            print(f"正确答案: {correct_answer}")
            print(f"是否正确: {'✓' if question_result['is_correct'] else '✗'}")
            print(f"置信度: {result['confidence']}")
            print("发现的因果路径:")
            for opt, paths in result['causal_paths'].items():
                if paths:
                    print(f"\n选项{opt}的路径:")
                    for path in paths:
                        print(f"- {path}")

            # 添加延时避免API限制
            time.sleep(1)

        return results

    def print_summary(self, results: List[Dict]):
        """打印评估总结"""
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = (correct / total) * 100

        print("\n=== 评估总结 ===")
        print(f"总题数: {total}")
        print(f"正确数: {correct}")
        print(f"准确率: {accuracy:.2f}%")

        print("\n=== 错误题目详情 ===")
        incorrect = [r for r in results if not r['is_correct']]
        for q in incorrect:
            print(f"\n问题 {q['question_number']}:")
            print(f"问题: {q['question']}")
            print(f"GPT答案: {q['gpt_answer']}")
            print(f"正确答案: {q['correct_answer']}")
            print(f"置信度: {q['confidence']}")
            print("选项:")
            for opt, text in q['options'].items():
                print(f"{opt}. {text}")
            print(f"分析: {q['analysis']}")
            print("\n因果路径:")
            for opt, paths in q['causal_paths'].items():
                if paths:
                    print(f"\n选项{opt}的路径:")
                    for path in paths:
                        print(f"- {path}")


def main():
    # 配置参数
    neo4j_params = {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "luohang819",
        "database": "casual-1"
    }
    openai_api_key = "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"  # 替换成你的实际API密钥


    # 创建评估器
    evaluator = CausalPathQAEvaluator(openai_api_key, neo4j_params)

    # 加载数据
    df, cui_data = evaluator.load_data(
        '../data/testdata/samples.csv',
        '../data/testdata/small_sample_cui.json'
    )

    # 评估问题
    results = evaluator.evaluate_questions(df, cui_data)

    # 打印总结
    evaluator.print_summary(results)

    # 保存结果
    with open('../../data/results/middleResults/causal-1_path_evaluation_normal_casual_graph_mini4.0.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

