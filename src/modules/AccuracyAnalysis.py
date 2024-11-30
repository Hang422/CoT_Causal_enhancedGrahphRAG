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
        self.logger = config.get_logger("question_analyzer")
        self.cache_root = config.paths["cache"] / path
        self.stages = ['derelict', 'casual', 'enhancement', 'final']  # 添加 enhancement

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
                'has_kg_paths_count': 0,
                'has_enhancement_count': 0
            }

        total = len(questions)
        correct = sum(1 for q in questions if q.is_correct)

        confidences = [q.confidence for q in questions if q.confidence is not None]
        confidence_count = len(confidences)

        confidence_ranges = {
            '0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0
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

        has_reasoning = sum(1 for q in questions if q.reasoning is not None)
        has_casual_paths = sum(
            1 for q in questions if any(q.casual_paths.get(opt) for opt in ['opa', 'opb', 'opc', 'opd']))
        has_kg_paths = sum(1 for q in questions if q.KG_paths and len(q.KG_paths) > 0)
        has_enhancement = sum(1 for q in questions if hasattr(q, 'integrated_paths') and q.integrated_paths)

        return {
            'total_questions': total,
            'correct_count': correct,
            'accuracy': correct / total if total > 0 else 0,
            'average_confidence': sum(confidences) / confidence_count if confidence_count > 0 else 0,
            'questions_with_confidence': confidence_count,
            'confidence_distribution': {k: v / confidence_count if confidence_count > 0 else 0 for k, v in
                                        confidence_ranges.items()},
            'has_reasoning_count': has_reasoning,
            'has_casual_paths_count': has_casual_paths,
            'has_kg_paths_count': has_kg_paths,
            'has_enhancement_count': has_enhancement
        }

    def analyze_all_stages(self) -> Dict[str, Dict]:
        """分析所有阶段的结果"""
        results = {}
        for stage in self.stages:
            questions = self.load_stage_questions(stage)
            results[stage] = self.analyze_stage(questions)
        return results

    def generate_report(self) -> pd.DataFrame:
        results = self.analyze_all_stages()
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
                'With KG Paths': f"{stats['has_kg_paths_count']}/{stats['total_questions']}",
                'With Enhancement': f"{stats['has_enhancement_count']}/{stats['total_questions']}"
            }

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


from pathlib import Path
from typing import Dict, List, Optional
import json
from src.modules.MedicalQuestion import MedicalQuestion
from config import config
import logging


def analyze_empty_answers(path: str) -> Dict:
    """分析 enhancement 阶段中的空答案情况"""
    logger = config.get_logger("empty_answer_analyzer")
    cache_root = config.paths["cache"] / path
    stage_path = cache_root / 'enhancement'

    if not stage_path.exists():
        logger.warning(f"Enhancement stage directory not found: {stage_path}")
        return {}

    # 统计结果
    stats = {
        'total_questions': 0,
        'empty_answers': 0,
        'empty_answer_details': [],
        'has_paths_empty': 0,  # 有路径但答案为空的数量
        'no_paths_empty': 0  # 无路径且答案为空的数量
    }

    # 加载并分析问题
    for cache_file in stage_path.glob("*.json"):
        try:
            with cache_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
                question = MedicalQuestion(**data)

            stats['total_questions'] += 1

            if not question.answer:  # 答案为空
                stats['empty_answers'] += 1

                # 检查是否有路径
                has_casual_paths = (hasattr(question, 'casual_paths_nodes_refine') and
                                    question.casual_paths_nodes_refine.get('start') and
                                    question.casual_paths_nodes_refine.get('end'))

                has_kg_paths = (hasattr(question, 'entities_original_pairs') and
                                question.entities_original_pairs.get('start') and
                                question.entities_original_pairs.get('end'))

                # 记录详细信息
                detail = {
                    'question': question.question,
                    'correct_answer': question.correct_answer,
                    'has_casual_paths': has_casual_paths,
                    'has_kg_paths': has_kg_paths
                }
                stats['empty_answer_details'].append(detail)

                # 统计有无路径的空答案数量
                if has_casual_paths or has_kg_paths:
                    stats['has_paths_empty'] += 1
                else:
                    stats['no_paths_empty'] += 1

        except Exception as e:
            logger.error(f"Error processing {cache_file}: {str(e)}")
            continue

    # 计算百分比
    if stats['total_questions'] > 0:
        stats['empty_answer_percentage'] = (stats['empty_answers'] / stats['total_questions']) * 100

    if stats['empty_answers'] > 0:
        stats['has_paths_empty_percentage'] = (stats['has_paths_empty'] / stats['empty_answers']) * 100
        stats['no_paths_empty_percentage'] = (stats['no_paths_empty'] / stats['empty_answers']) * 100

    return stats


def print_empty_answer_analysis(path: str):
    """打印空答案分析结果"""
    stats = analyze_empty_answers(path)

    if not stats:
        print("No enhancement stage data found.")
        return

    print("\nEmpty Answer Analysis for Enhancement Stage:")
    print(f"Total questions analyzed: {stats['total_questions']}")
    print(f"Questions with empty answers: {stats['empty_answers']} ({stats.get('empty_answer_percentage', 0):.2f}%)")
    print(f"\nAmong empty answers:")
    print(f"- With paths: {stats['has_paths_empty']} ({stats.get('has_paths_empty_percentage', 0):.2f}%)")
    print(f"- Without paths: {stats['no_paths_empty']} ({stats.get('no_paths_empty_percentage', 0):.2f}%)")

    print("\nDetailed list of questions with empty answers:")
    for idx, detail in enumerate(stats['empty_answer_details'], 1):
        print(f"\n{idx}. Question: {detail['question']}")
        print(f"   Correct Answer: {detail['correct_answer']}")
        print(f"   Has Casual Paths: {'Yes' if detail['has_casual_paths'] else 'No'}")
        print(f"   Has KG Paths: {'Yes' if detail['has_kg_paths'] else 'No'}")


def main():
    path = '20-gpt-4-adaptive-knowledge-0.75-shortest-enhance-ultra2'
    analyzer = QuestionAnalyzer(path)
    analyzer.save_report(f"{path}/report.xlsx")
    analyse(path)


if __name__ == "__main__":
    path = '20-gpt-4-adaptive-knowledge-0.75-shortest-enhance-ultra1'
    path = '20-gpt-4-adaptive-knowledge-0.75-shortest-enhance-ultra2'
    print_empty_answer_analysis(path)
    main()
