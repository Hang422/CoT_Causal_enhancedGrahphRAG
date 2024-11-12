import json
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re


class AccuracyCalculator:
    def __init__(self, predictions_file: str, answers_file: str):
        """
        初始化准确率计算器

        Args:
            predictions_file: 模型预测结果的JSON文件路径
            answers_file: 正确答案的CSV文件路径
        """
        self.predictions = self._load_predictions(predictions_file)
        self.answers = self._load_answers(answers_file)

    def _load_predictions(self, file_path: str) -> List[Dict]:
        """加载模型预测结果"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_answers(self, file_path: str) -> pd.DataFrame:
        """加载正确答案"""
        return pd.read_csv(file_path)

    def _convert_cop_to_letter(self, cop: int) -> str:
        """将数字答案转换为字母"""
        return chr(97 + cop)  # 0->a, 1->b, etc.

    def _extract_answer_letter(self, answer: str) -> str:
        """
        从答案字符串中提取选项字母

        Args:
            answer: 答案字符串，可能包含额外文本，如 "a (ventricular bigeminy)"

        Returns:
            str: 提取出的选项字母
        """
        # 移除所有空白字符
        answer = answer.strip().lower()

        # 如果答案为空，返回空字符串
        if not answer:
            return ""

        # 使用正则表达式匹配答案开头的字母
        match = re.match(r'^([abcd])', answer)
        if match:
            return match.group(1)

        # 如果没有找到有效的选项字母，返回原始答案的第一个字符
        return answer[0] if answer else ""

    def calculate_metrics(self) -> Dict:
        """
        计算各种评估指标

        Returns:
            Dict: 包含准确率、置信度分析等指标的字典
        """
        total = len(self.predictions)
        correct = 0
        confidence_data = []
        correct_confidence = []
        incorrect_confidence = []

        for i, pred in enumerate(self.predictions):
            # 获取正确答案
            correct_answer = self._convert_cop_to_letter(self.answers.iloc[i]['cop'])

            # 从预测答案中提取选项字母
            predicted_answer = self._extract_answer_letter(pred['answer'])
            confidence = pred['confidence']

            # 统计正确率
            is_correct = predicted_answer == correct_answer
            if is_correct:
                correct += 1
                correct_confidence.append(confidence)
            else:
                incorrect_confidence.append(confidence)

            confidence_data.append({
                'confidence': confidence,
                'is_correct': is_correct,
                'predicted': predicted_answer,
                'actual': correct_answer,
                'full_answer': pred['answer']  # 保存完整答案以供分析
            })

        # 计算指标
        accuracy = (correct / total) * 100
        avg_confidence = sum(pred['confidence'] for pred in self.predictions) / total
        avg_correct_conf = sum(correct_confidence) / len(correct_confidence) if correct_confidence else 0
        avg_incorrect_conf = sum(incorrect_confidence) / len(incorrect_confidence) if incorrect_confidence else 0

        return {
            'total_questions': total,
            'correct_answers': correct,
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'average_confidence_correct': avg_correct_conf,
            'average_confidence_incorrect': avg_incorrect_conf,
            'confidence_data': confidence_data
        }

    def generate_report(self) -> str:
        """生成详细的评估报告"""
        metrics = self.calculate_metrics()

        report = "=== 模型评估报告 ===\n\n"
        report += f"总题数: {metrics['total_questions']}\n"
        report += f"正确题数: {metrics['correct_answers']}\n"
        report += f"准确率: {metrics['accuracy']:.2f}%\n\n"

        report += "置信度分析:\n"
        report += f"平均置信度: {metrics['average_confidence']:.2f}\n"
        report += f"正确答案的平均置信度: {metrics['average_confidence_correct']:.2f}\n"
        report += f"错误答案的平均置信度: {metrics['average_confidence_incorrect']:.2f}\n\n"

        # 添加错误分析
        report += "错误答案分析:\n"
        for i, pred in enumerate(self.predictions):
            correct_answer = self._convert_cop_to_letter(self.answers.iloc[i]['cop'])
            predicted_answer = self._extract_answer_letter(pred['answer'])

            if predicted_answer != correct_answer:
                report += f"\n问题 {i + 1}:\n"
                report += f"预测答案: {pred['answer']}\n"  # 显示完整预测答案
                report += f"正确答案: {correct_answer}\n"
                report += f"置信度: {pred['confidence']}\n"
                if pred.get('final_analysis'):
                    report += f"分析: {pred['final_analysis'][:200]}...\n"  # 截取前200个字符

        return report

    def plot_confidence_distribution(self):
        """绘制置信度分布图"""
        metrics = self.calculate_metrics()
        confidence_data = metrics['confidence_data']

        correct_conf = [d['confidence'] for d in confidence_data if d['is_correct']]
        incorrect_conf = [d['confidence'] for d in confidence_data if not d['is_correct']]

        plt.figure(figsize=(10, 6))
        plt.hist([correct_conf, incorrect_conf], label=['正确', '错误'], bins=10, alpha=0.7)
        plt.xlabel('置信度')
        plt.ylabel('频次')
        plt.title('答案正确性与置信度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)

        return plt

    def get_confusion_matrix(self):
        """
        生成混淆矩阵
        """
        y_true = []
        y_pred = []

        for i, pred in enumerate(self.predictions):
            correct_answer = self._convert_cop_to_letter(self.answers.iloc[i]['cop'])
            predicted_answer = self._extract_answer_letter(pred['answer'])

            # 将字母转换为数字索引
            y_true.append(ord(correct_answer) - ord('a'))
            y_pred.append(ord(predicted_answer) - ord('a'))

        return confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(self):
        """
        绘制混淆矩阵热力图
        """
        cm = self.get_confusion_matrix()
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['A', 'B', 'C', 'D'],
                    yticklabels=['A', 'B', 'C', 'D'])
        plt.title('答案选项混淆矩阵')
        plt.xlabel('预测答案')
        plt.ylabel('实际答案')
        return plt


def main():
    # 初始化计算器
    calculator = AccuracyCalculator(
        predictions_file='../../data/results/middleResults/final_answers_kg.json',
        answers_file='../../data/testdata/samples.csv'
    )

    # 生成报告
    report = calculator.generate_report()
    print(report)

    # 保存报告
    with open('evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # 绘制置信度分布图
    plt = calculator.plot_confidence_distribution()
    plt.savefig('confidence_distribution.png')
    plt.close()

    # 绘制混淆矩阵
    plt = calculator.plot_confusion_matrix()
    plt.savefig('confusion_matrix.png')
    plt.close()


if __name__ == '__main__':
    main()