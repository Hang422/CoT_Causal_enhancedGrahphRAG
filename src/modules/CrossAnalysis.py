from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd
from src.modules.MedicalQuestion import MedicalQuestion
from config import config
import logging


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

    def get_cross_model_analysis(self) -> Dict:
        """对三个模型进行交叉分析"""
        # 加载所有阶段的问题
        baseline_questions = {q.question: q for q in self.load_stage_questions('derelict')}
        casual_questions = {q.question: q for q in self.load_stage_questions('casual')}
        final_questions = {q.question: q for q in self.load_stage_questions('final')}

        # 初始化结果统计
        analysis = {
            'baseline_correct': {
                'casual_model': {'with_path': 0, 'without_path': 0, 'total': 0},
                'final_model': {'with_path': 0, 'without_path': 0, 'total': 0},
            },
            'baseline_incorrect': {
                'casual_model': {'with_path': 0, 'without_path': 0, 'total': 0},
                'final_model': {'with_path': 0, 'without_path': 0, 'total': 0},
            },
            'counts': {
                'baseline_correct': 0,
                'baseline_incorrect': 0
            }
        }

        # 遍历基线模型的所有问题
        for question, baseline_q in baseline_questions.items():
            casual_q = casual_questions.get(question)
            final_q = final_questions.get(question)

            if not all([casual_q, final_q]):
                continue

            # 确定是基线正确还是错误的case
            if baseline_q.is_correct:
                analysis['counts']['baseline_correct'] += 1

                # 分析casual模型表现
                if casual_q.is_correct:
                    if casual_q.has_complete_paths():
                        analysis['baseline_correct']['casual_model']['with_path'] += 1
                    else:
                        analysis['baseline_correct']['casual_model']['without_path'] += 1
                analysis['baseline_correct']['casual_model']['total'] += 1

                # 分析final模型表现
                if final_q.is_correct:
                    if final_q.has_complete_paths():
                        analysis['baseline_correct']['final_model']['with_path'] += 1
                    else:
                        analysis['baseline_correct']['final_model']['without_path'] += 1
                analysis['baseline_correct']['final_model']['total'] += 1

            else:
                analysis['counts']['baseline_incorrect'] += 1

                # 分析casual模型表现
                if casual_q.is_correct:
                    if casual_q.has_complete_paths():
                        analysis['baseline_incorrect']['casual_model']['with_path'] += 1
                    else:
                        analysis['baseline_incorrect']['casual_model']['without_path'] += 1
                analysis['baseline_incorrect']['casual_model']['total'] += 1

                # 分析final模型表现
                if final_q.is_correct:
                    if final_q.has_complete_paths():
                        analysis['baseline_incorrect']['final_model']['with_path'] += 1
                    else:
                        analysis['baseline_incorrect']['final_model']['without_path'] += 1
                analysis['baseline_incorrect']['final_model']['total'] += 1

        # 计算百分比
        for baseline_result in ['baseline_correct', 'baseline_incorrect']:
            total = analysis['counts'][baseline_result]
            if total > 0:
                for model in ['casual_model', 'final_model']:
                    model_total = analysis[baseline_result][model]['total']
                    if model_total > 0:
                        with_path = analysis[baseline_result][model]['with_path']
                        without_path = analysis[baseline_result][model]['without_path']

                        # 转换为百分比
                        analysis[baseline_result][model]['with_path'] = (with_path / total) * 100
                        analysis[baseline_result][model]['without_path'] = (without_path / total) * 100

        return analysis

    def generate_visualization_data(self) -> Dict:
        """生成可视化所需的数据"""
        analysis = self.get_cross_model_analysis()

        return {
            'baselineCorrect': [
                {
                    'model': 'Causal Model',
                    'withPath': round(analysis['baseline_correct']['casual_model']['with_path'], 1),
                    'withoutPath': round(analysis['baseline_correct']['casual_model']['without_path'], 1),
                },
                {
                    'model': 'Final Model',
                    'withPath': round(analysis['baseline_correct']['final_model']['with_path'], 1),
                    'withoutPath': round(analysis['baseline_correct']['final_model']['without_path'], 1),
                }
            ],
            'baselineIncorrect': [
                {
                    'model': 'Causal Model',
                    'withPath': round(analysis['baseline_incorrect']['casual_model']['with_path'], 1),
                    'withoutPath': round(analysis['baseline_incorrect']['casual_model']['without_path'], 1),
                },
                {
                    'model': 'Final Model',
                    'withPath': round(analysis['baseline_incorrect']['final_model']['with_path'], 1),
                    'withoutPath': round(analysis['baseline_incorrect']['final_model']['without_path'], 1),
                }
            ],
            'counts': analysis['counts']
        }

    def save_analysis(self, file_name,output_path: Optional[str] = None) -> None:
        """保存分析结果"""
        analysis_data = self.get_cross_model_analysis()

        if output_path is None:
            output_path = config.paths["output"] / file_name
        else:
            output_path = Path(output_path)

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存为JSON文件
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Analysis results saved to {output_path}")


def analyse(path):
    analyzer = QuestionAnalyzer(path)

    # 生成分析数据
    vis_data = analyzer.generate_visualization_data()

    # 打印分析结果
    print("\nAnalysis Results:")
    print(f"Total questions answered correctly by baseline: {vis_data['counts']['baseline_correct']}")
    print(f"Total questions answered incorrectly by baseline: {vis_data['counts']['baseline_incorrect']}")

    print("\nPerformance on Baseline Correct Questions:")
    for model in vis_data['baselineCorrect']:
        print(f"\n{model['model']}:")
        print(f"  Correct with path: {model['withPath']}%")
        print(f"  Correct without path: {model['withoutPath']}%")
        print(f"  Total correct: {model['withPath'] + model['withoutPath']}%")

    print("\nPerformance on Baseline Incorrect Questions:")
    for model in vis_data['baselineIncorrect']:
        print(f"\n{model['model']}:")
        print(f"  Correct with path: {model['withPath']}%")
        print(f"  Correct without path: {model['withoutPath']}%")
        print(f"  Total correct: {model['withPath'] + model['withoutPath']}%")

    # 保存分析结果
    analyzer.save_analysis(f"{path}/analysis.json")


if __name__ == "__main__":
    analyse('20-4o-casual-knowledge-0.6-shortest')