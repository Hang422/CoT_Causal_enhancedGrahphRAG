import pandas as pd
from typing import Dict, List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import json
from get_answer import MedicalQAGPT

class QAEvaluator:
    def __init__(self, api_key: str):
        self.qa_gpt = MedicalQAGPT(api_key)

    def load_questions(self, csv_path: str) -> pd.DataFrame:
        """加载CSV文件中的问题"""
        df = pd.read_csv(csv_path)
        return df

    def format_question_data(self, row) -> tuple:
        """格式化单个问题的数据"""
        question = row['question']
        options = {
            'a': row['opa'],
            'b': row['opb'],
            'c': row['opc'],
            'd': row['opd']
        }
        correct_answer = chr(97 + int(row['cop']))  # 0->a, 1->b, 2->c, 3->d
        return question, options, correct_answer

    def clean_answer(self, answer: str) -> str:
        """清理GPT返回的答案，只保留选项字母"""
        if not answer:
            return None
        # 提取第一个字母并转为小写
        answer = answer.strip().lower()
        return answer[0] if answer and answer[0] in 'abcd' else None

    def evaluate_questions(self, df: pd.DataFrame) -> List[Dict]:
        """评估所有问题"""
        results = []

        for index, row in df.iterrows():
            question, options, correct_answer = self.format_question_data(row)

            print(f"\n处理问题 {index + 1}/{len(df)}:")
            print(f"问题: {question}")

            # 获取不使用因果图谱的答案
            result = self.qa_gpt.get_answer_without_paths(question, options)

            # 清理GPT的答案
            cleaned_answer = self.clean_answer(result['answer'])

            # 构建结果
            question_result = {
                'question_number': index + 1,
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
                'gpt_raw_answer': result['answer'],
                'gpt_answer': cleaned_answer,
                'is_correct': cleaned_answer == correct_answer if cleaned_answer else False,
                'confidence': result['confidence'],
                'analysis': result['analysis']
            }

            results.append(question_result)

            # 打印当前结果
            print(f"GPT原始答案: {result['answer']}")
            print(f"GPT处理后答案: {cleaned_answer}")
            print(f"正确答案: {correct_answer}")
            print(f"是否正确: {'✓' if question_result['is_correct'] else '✗'}")
            print(f"置信度: {result['confidence']}")

            # 添加延时避免API限制
            time.sleep(1)

        return results

    def get_incorrect_answers(self, results: List[Dict]) -> List[Dict]:
        """获取答错的题目"""
        return [r for r in results if not r['is_correct']]

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
        incorrect = self.get_incorrect_answers(results)
        for q in incorrect:
            print(f"\n问题 {q['question_number']}:")
            print(f"问题: {q['question']}")
            print(f"GPT原始答案: {q['gpt_raw_answer']}")
            print(f"GPT处理后答案: {q['gpt_answer']}")
            print(f"正确答案: {q['correct_answer']}")
            print(f"置信度: {q['confidence']}")
            print("选项:")
            for opt, text in q['options'].items():
                print(f"{opt}. {text}")
            print(f"GPT分析: {q['analysis']}")


def main():
    api_key = "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"  # 替换成你的实际API密钥
    evaluator = QAEvaluator(api_key)

    # 加载并评估题目
    df = evaluator.load_questions('../data/testdata/samples.csv')
    results = evaluator.evaluate_questions(df)

    # 打印总结
    evaluator.print_summary(results)

    # 保存结果到文件
    with open('../../data/results/middleResults/mini4.0_ormal_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()