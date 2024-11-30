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
        self.logger = config.get_logger("question_analyzer")
        self.cache_root = config.paths["cache"] / path
        self.stages = ['derelict', 'casual', 'enhancement', 'final']

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
        """对模型进行交叉分析，包含CG、KG和Enhanced Information的分析"""
        baseline_questions = {q.question: q for q in self.load_stage_questions('derelict')}
        casual_questions = {q.question: q for q in self.load_stage_questions('casual')}
        enhancement_questions = {q.question: q for q in self.load_stage_questions('enhancement')}
        final_questions = {q.question: q for q in self.load_stage_questions('final')}

        analysis = {
            'baseline_correct': {
                'casual_model': {'with_cg': 0, 'with_kg': 0, 'with_both': 0, 'without_paths': 0, 'total': 0},
                'enhancement_model': {'with_cg': 0, 'with_kg': 0, 'with_both': 0, 'without_paths': 0, 'total': 0},
                'final_model': {'with_cg': 0, 'with_kg': 0, 'with_both': 0, 'without_paths': 0, 'total': 0}
            },
            'baseline_incorrect': {
                'casual_model': {'with_cg': 0, 'with_kg': 0, 'with_both': 0, 'without_paths': 0, 'total': 0},
                'enhancement_model': {'with_cg': 0, 'with_kg': 0, 'with_both': 0, 'without_paths': 0, 'total': 0},
                'final_model': {'with_cg': 0, 'with_kg': 0, 'with_both': 0, 'without_paths': 0, 'total': 0}
            },
            'counts': {
                'baseline_correct': 0,
                'baseline_incorrect': 0
            }
        }

        def check_paths(question):
            has_cg = any(question.casual_paths.get(opt) for opt in ['opa', 'opb', 'opc', 'opd'])
            has_kg = bool(question.KG_paths and len(question.KG_paths) > 0)
            return has_cg, has_kg

        def check_enhancement_paths(question):
            # Check CG through casual_paths_nodes_refine
            has_cg = (hasattr(question, 'casual_paths_nodes_refine') and
                      question.casual_paths_nodes_refine.get('start') and
                      question.casual_paths_nodes_refine.get('end'))

            # Check KG through entities_original_pairs
            has_kg = (hasattr(question, 'entities_original_pairs') and
                      question.entities_original_pairs.get('start') and
                      question.entities_original_pairs.get('end'))

            return has_cg, has_kg

        for question, baseline_q in baseline_questions.items():
            casual_q = casual_questions.get(question)
            enhancement_q = enhancement_questions.get(question)
            final_q = final_questions.get(question)

            if not all([casual_q, enhancement_q, final_q]):
                continue

            target_category = 'baseline_correct' if baseline_q.is_correct else 'baseline_incorrect'
            analysis['counts'][target_category] += 1

            # Analyze each model
            for model_type, model_q in [('casual_model', casual_q),
                                        ('enhancement_model', enhancement_q),
                                        ('final_model', final_q)]:
                if model_q.is_correct:
                    # Use different path checking for enhancement model
                    if model_type == 'enhancement_model':
                        has_cg, has_kg = check_enhancement_paths(model_q)
                    else:
                        has_cg, has_kg = check_paths(model_q)

                    if has_cg and has_kg:
                        analysis[target_category][model_type]['with_both'] += 1
                    elif has_cg:
                        analysis[target_category][model_type]['with_cg'] += 1
                    elif has_kg:
                        analysis[target_category][model_type]['with_kg'] += 1
                    else:
                        analysis[target_category][model_type]['without_paths'] += 1
                analysis[target_category][model_type]['total'] += 1

        # Calculate percentages
        for baseline_result in ['baseline_correct', 'baseline_incorrect']:
            total = analysis['counts'][baseline_result]
            if total > 0:
                for model in ['casual_model', 'enhancement_model', 'final_model']:
                    for key in ['with_cg', 'with_kg', 'with_both', 'without_paths']:
                        current_value = analysis[baseline_result][model][key]
                        analysis[baseline_result][model][key] = (current_value / total) * 100

        return analysis

    def generate_visualization_data(self) -> Dict:
        """生成可视化所需的数据"""
        analysis = self.get_cross_model_analysis()

        return {
            'baselineCorrect': [
                {
                    'model': 'Causal Model',
                    'withCG': round(analysis['baseline_correct']['casual_model']['with_cg'], 1),
                    'withKG': round(analysis['baseline_correct']['casual_model']['with_kg'], 1),
                    'withBoth': round(analysis['baseline_correct']['casual_model']['with_both'], 1),
                    'withoutPaths': round(analysis['baseline_correct']['casual_model']['without_paths'], 1),
                },
                {
                    'model': 'Enhancement Model',
                    'withCG': round(analysis['baseline_correct']['enhancement_model']['with_cg'], 1),
                    'withKG': round(analysis['baseline_correct']['enhancement_model']['with_kg'], 1),
                    'withBoth': round(analysis['baseline_correct']['enhancement_model']['with_both'], 1),
                    'withoutPaths': round(analysis['baseline_correct']['enhancement_model']['without_paths'], 1),
                },
                {
                    'model': 'Final Model',
                    'withCG': round(analysis['baseline_correct']['final_model']['with_cg'], 1),
                    'withKG': round(analysis['baseline_correct']['final_model']['with_kg'], 1),
                    'withBoth': round(analysis['baseline_correct']['final_model']['with_both'], 1),
                    'withoutPaths': round(analysis['baseline_correct']['final_model']['without_paths'], 1),
                }
            ],
            'baselineIncorrect': [
                {
                    'model': 'Causal Model',
                    'withCG': round(analysis['baseline_incorrect']['casual_model']['with_cg'], 1),
                    'withKG': round(analysis['baseline_incorrect']['casual_model']['with_kg'], 1),
                    'withBoth': round(analysis['baseline_incorrect']['casual_model']['with_both'], 1),
                    'withoutPaths': round(analysis['baseline_incorrect']['casual_model']['without_paths'], 1),
                },
                {
                    'model': 'Enhancement Model',
                    'withCG': round(analysis['baseline_incorrect']['enhancement_model']['with_cg'], 1),
                    'withKG': round(analysis['baseline_incorrect']['enhancement_model']['with_kg'], 1),
                    'withBoth': round(analysis['baseline_incorrect']['enhancement_model']['with_both'], 1),
                    'withoutPaths': round(analysis['baseline_incorrect']['enhancement_model']['without_paths'], 1),
                },
                {
                    'model': 'Final Model',
                    'withCG': round(analysis['baseline_incorrect']['final_model']['with_cg'], 1),
                    'withKG': round(analysis['baseline_incorrect']['final_model']['with_kg'], 1),
                    'withBoth': round(analysis['baseline_incorrect']['final_model']['with_both'], 1),
                    'withoutPaths': round(analysis['baseline_incorrect']['final_model']['without_paths'], 1),
                }
            ],
            'counts': analysis['counts']
        }

    def save_analysis(self, file_name, output_path: Optional[str] = None) -> None:
        """保存分析结果"""
        analysis_data = self.get_cross_model_analysis()

        if output_path is None:
            output_path = config.paths["output"] / file_name
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Analysis results saved to {output_path}")


def analyse(path):
    analyzer = QuestionAnalyzer(path)
    vis_data = analyzer.generate_visualization_data()

    print("\nAnalysis Results:")
    print(f"Total questions answered correctly by baseline: {vis_data['counts']['baseline_correct']}")
    print(f"Total questions answered incorrectly by baseline: {vis_data['counts']['baseline_incorrect']}")

    print("\nPerformance on Baseline Correct Questions:")
    for model in vis_data['baselineCorrect']:
        print(f"\n{model['model']}:")
        print(f"  Correct with CG only: {model['withCG']}%")
        print(f"  Correct with KG only: {model['withKG']}%")
        print(f"  Correct with both CG and KG: {model['withBoth']}%")
        print(f"  Correct without paths: {model['withoutPaths']}%")
        print(f"  Total correct: {model['withCG'] + model['withKG'] + model['withBoth'] + model['withoutPaths']}%")

    print("\nPerformance on Baseline Incorrect Questions:")
    for model in vis_data['baselineIncorrect']:
        print(f"\n{model['model']}:")
        print(f"  Correct with CG only: {model['withCG']}%")
        print(f"  Correct with KG only: {model['withKG']}%")
        print(f"  Correct with both CG and KG: {model['withBoth']}%")
        print(f"  Correct without paths: {model['withoutPaths']}%")
        print(f"  Total correct: {model['withCG'] + model['withKG'] + model['withBoth'] + model['withoutPaths']}%")

    analyzer.save_analysis(f"{path}/analysis.json")


if __name__ == "__main__":
    analyse('20-gpt-4o-adaptive-knowledge-0.75-shortest-enhance')