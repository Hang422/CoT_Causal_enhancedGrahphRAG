from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd
from src.modules.MedicalQuestion import MedicalQuestion
from config import config
import logging
from src.modules.CrossAnalysis import analyse

class QuestionAnalyzer:
    """分析不同阶段问题处理结果的分析器"""

    def __init__(self, path):
        """初始化分析器"""
        self.logger = config.get_logger("question_analyzer")
        self.cache_root = config.paths["cache"] / path
        self.stages = ['derelict', 'casual', 'final']

    def load_stage_questions(self, stage: str) -> List[MedicalQuestion]:
        """加载某个阶段的所有问题"""
        questions = []
        stage_path = self.cache_root / stage

        if not stage_path.exists():
            self.logger.warning(f"Stage directory not found: {stage_path}")
            return questions

        for cache_file in stage_path.glob("*.json"):
            try:
                with cache_file.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                    question = MedicalQuestion(**data)
                    questions.append(question)
            except Exception as e:
                self.logger.error(f"Error loading {cache_file}: {str(e)}")
                continue

        return questions

    def analyze_stage(self, questions: List[MedicalQuestion]) -> Dict:
        """分析某个阶段的问题集合"""
        if not questions:
            return {
                'total_questions': 0,
                'correct_count': 0,
                'accuracy': 0,
                'average_confidence': 0,
                'questions_with_confidence': 0,
                'confidence_distribution': {},
                'has_reasoning_count': 0,
                'has_casual_paths_count': 0,
                'has_kg_paths_count': 0
            }

        total = len(questions)
        correct = sum(1 for q in questions if q.is_correct)

        # 置信度分析
        confidences = [q.confidence for q in questions if q.confidence is not None]
        confidence_count = len(confidences)

        # 置信度分布
        confidence_ranges = {
            '0-0.2': 0,
            '0.2-0.4': 0,
            '0.4-0.6': 0,
            '0.6-0.8': 0,
            '0.8-1.0': 0
        }

        for conf in confidences:
            if 0 <= conf < 0.2:
                confidence_ranges['0-0.2'] += 1
            elif 0.2 <= conf < 0.4:
                confidence_ranges['0.2-0.4'] += 1
            elif 0.4 <= conf < 0.6:
                confidence_ranges['0.4-0.6'] += 1
            elif 0.6 <= conf < 0.8:
                confidence_ranges['0.6-0.8'] += 1
            else:
                confidence_ranges['0.8-1.0'] += 1

        # 路径和推理分析
        has_reasoning = sum(1 for q in questions if q.reasoning is not None)
        has_casual_paths = sum(1 for q in questions if any(q.casual_paths.get(opt) for opt in ['opa', 'opb', 'opc', 'opd']))
        has_kg_paths = sum(1 for q in questions if q.KG_paths and len(q.KG_paths) > 0)

        return {
            'total_questions': total,
            'correct_count': correct,
            'accuracy': correct / total if total > 0 else 0,
            'average_confidence': sum(confidences) / confidence_count if confidence_count > 0 else 0,
            'questions_with_confidence': confidence_count,
            'confidence_distribution': {
                k: v / confidence_count if confidence_count > 0 else 0
                for k, v in confidence_ranges.items()
            },
            'has_reasoning_count': has_reasoning,
            'has_casual_paths_count': has_casual_paths,
            'has_kg_paths_count': has_kg_paths
        }

    def analyze_all_stages(self) -> Dict[str, Dict]:
        """分析所有阶段的结果"""
        results = {}
        for stage in self.stages:
            questions = self.load_stage_questions(stage)
            results[stage] = self.analyze_stage(questions)
        return results

    def generate_report(self) -> pd.DataFrame:
        """生成分析报告"""
        results = self.analyze_all_stages()

        # 创建比较表格
        comparison_data = []

        for stage, stats in results.items():
            row = {
                'Stage': stage,
                'Total Questions': stats['total_questions'],
                'Correct Count': stats['correct_count'],
                'Accuracy': f"{stats['accuracy']:.2%}",
                'Avg Confidence': f"{stats['average_confidence']:.2%}",
                'With Confidence': f"{stats['questions_with_confidence']}/{stats['total_questions']}",
                'With Reasoning': f"{stats['has_reasoning_count']}/{stats['total_questions']}",
                'With Casual Paths': f"{stats['has_casual_paths_count']}/{stats['total_questions']}",
                'With KG Paths': f"{stats['has_kg_paths_count']}/{stats['total_questions']}"
            }

            # 添加置信度分布
            for range_name, percentage in stats['confidence_distribution'].items():
                row[f'Conf {range_name}'] = f"{percentage:.2%}"

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def save_report(self, file_name, output_path: Optional[str] = None) -> None:
        """保存分析报告"""
        df = self.generate_report()

        if output_path is None:
            output_path = config.paths["output"] / file_name
        else:
            output_path = Path(output_path)

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存为Excel文件
        df.to_excel(output_path, index=False)
        self.logger.info(f"Analysis report saved to {output_path}")

        # 打印到控制台
        print("\nAnalysis Report:")
        print(df.to_string())


def main():
    path = '40-4omini-adaptive-knowledge-0.75-shortest'
    analyzer = QuestionAnalyzer(path)
    analyzer.save_report(f"{path}/report.xlsx")
    analyse(path)

if __name__ == "__main__":
    main()