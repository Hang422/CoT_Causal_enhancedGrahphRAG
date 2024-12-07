from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd
from src.modules.MedicalQuestion import MedicalQuestion
from config import config
import logging
from src.modules.CrossAnalysis import analyse


def _is_correct(question: Dict) -> bool:
    """Check if the answer matches correct_answer"""
    return question.get('answer') == question.get('correct_answer')


class CrossPathAnalyzer:
    def __init__(self, path):
        self.logger = config.get_logger("cross_path_analyzer")
        self.cache_root = config.paths["cache"] / path
        self.stages = ['derelict', 'enhanced', 'knowledge_graph',
                       'causal_graph', 'graph_enhanced', 'llm_enhanced']

    def _load_questions(self, stage: str) -> List[Dict]:
        """Load questions from JSON files"""
        questions = []
        stage_path = self.cache_root / stage
        if not stage_path.exists():
            self.logger.warning(f"Stage directory not found: {stage_path}")
            return questions

        for cache_file in stage_path.glob("*.json"):
            try:
                with cache_file.open('r', encoding='utf-8') as f:
                    question = json.load(f)
                    questions.append(question)
            except Exception as e:
                self.logger.error(f"Error loading {cache_file}: {str(e)}")
        return questions

    def analyze(self, stage: str) -> Dict:
        """Analyze path presence and accuracies comparing baseline and stage results"""
        baseline_questions = {q['question']: q for q in self._load_questions('derelict')}
        stage_questions = {q['question']: q for q in self._load_questions(stage)}

        results = {
            "baseline_correct": {
                "with_initial_cg": 0,
                "with_cg": 0,
                "with_kg": 0,
                "with_both": 0,
                "without_paths": 0,
                "total": 0
            },
            "baseline_incorrect": {
                "with_initial_cg": 0,
                "with_cg": 0,
                "with_kg": 0,
                "with_both": 0,
                "without_paths": 0,
                "total": 0
            },
            "path_accuracies": {
                "with_initial_cg": {"correct": 0, "total": 0},
                "with_cg": {"correct": 0, "total": 0},
                "with_kg": {"correct": 0, "total": 0},
                "with_both": {"correct": 0, "total": 0},
                "without_paths": {"correct": 0, "total": 0}
            }
        }

        baseline_correct_count = 0
        baseline_incorrect_count = 0

        for question in set(baseline_questions.keys()) & set(stage_questions.keys()):
            baseline_q = baseline_questions[question]
            stage_q = stage_questions[question]

            # Check baseline correctness
            baseline_correct = _is_correct(baseline_q)
            stage_correct = _is_correct(stage_q)

            # Determine path presence
            has_initial_causal = bool(stage_q['initial_causal_graph']['paths'] and
                                      len(stage_q['initial_causal_graph']['paths']) > 0 and
                                      stage_q['initial_causal_graph'][
                                          'paths'] != "There is no obvious causal relationship.")
            has_causal = bool(stage_q['causal_graph']['paths'] and
                              len(stage_q['causal_graph']['paths']) > 0 and
                              stage_q['causal_graph']['paths'] != "There is no obvious causal relationship.")
            has_knowledge = bool(stage_q['knowledge_graph']['paths'] and
                                 len(stage_q['knowledge_graph']['paths']) > 0)

            # Determine path category
            if has_causal and has_knowledge:
                path_category = "with_both"
            elif has_initial_causal:
                path_category = "with_initial_cg"
            elif has_causal:
                path_category = "with_cg"
            elif has_knowledge:
                path_category = "with_kg"
            else:
                path_category = "without_paths"

            # Update path accuracies
            results["path_accuracies"][path_category]["total"] += 1
            if stage_correct:
                results["path_accuracies"][path_category]["correct"] += 1

            # Update baseline categories
            if baseline_correct:
                baseline_correct_count += 1
                if stage_correct:
                    results["baseline_correct"][path_category] += 1
                    results["baseline_correct"]["total"] += 1
            else:
                baseline_incorrect_count += 1
                if stage_correct:
                    results["baseline_incorrect"][path_category] += 1
                    results["baseline_incorrect"]["total"] += 1

        # Calculate path accuracies
        for path_type in ["with_initial_cg", "with_cg", "with_kg", "with_both", "without_paths"]:
            total = results["path_accuracies"][path_type]["total"]
            if total > 0:
                correct = results["path_accuracies"][path_type]["correct"]
                results["path_accuracies"][path_type] = {"accuracy": (correct / total * 100), "total": total}

        # Calculate overall accuracy
        all_questions = self._load_questions(stage)
        correct_count = sum(1 for q in all_questions if _is_correct(q))
        results["overall_accuracy"] = (correct_count / len(all_questions) * 100) if all_questions else 0

        return results

    def print_analysis(self) -> None:
        """打印分析结果"""
        results = self.analyze_all_stages()

        for stage, analysis in results.items():
            print(f"\n=== {stage} Analysis ===")
            print(f"Overall Accuracy: {analysis['overall_accuracy']:.2f}%")

            print("\nPath Type Overall Accuracies:")
            for path_type, accuracy in analysis['path_overall_accuracies'].items():
                print(f"  {path_type}: {accuracy:.2f}%")

            print("\nBaseline Correct Questions:")
            for key in ["with_cg", "with_kg", "with_both", "without_paths"]:
                print(f"  {key}: {analysis['baseline_correct'][key]:.2f}%")
            print(f"  Total count: {analysis['baseline_correct']['total']}")

            print("\nBaseline Incorrect Questions:")
            for key in ["with_cg", "with_kg", "with_both", "without_paths"]:
                print(f"  {key}: {analysis['baseline_incorrect'][key]:.2f}%")
            print(f"  Total count: {analysis['baseline_incorrect']['total']}")

    def analyze_all_stages(self) -> Dict:
        """分析所有阶段"""
        analyses = {stage: self.analyze(stage) for stage in self.stages}

        # 保存到 output 目录
        output_dir = self.cache_root / str(config.openai.get("model"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 JSON 分析结果
        with open(output_dir / 'analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analyses, f, ensure_ascii=False, indent=2)

        # 创建 Excel 报告
        df_data = []
        for stage, analysis in analyses.items():
            row = {
                'Stage': stage,
                'Overall Accuracy': f"{analysis['overall_accuracy']:.2f}%",
                'Baseline Correct - with CG': f"{analysis['baseline_correct']['with_cg']:.2f}%",
                'Baseline Correct - with KG': f"{analysis['baseline_correct']['with_kg']:.2f}%",
                'Baseline Correct - with Both': f"{analysis['baseline_correct']['with_both']:.2f}%",
                'Baseline Correct - without Paths': f"{analysis['baseline_correct']['without_paths']:.2f}%",
                'Baseline Correct Total': analysis['baseline_correct']['total'],
                'Baseline Incorrect - with CG': f"{analysis['baseline_incorrect']['with_cg']:.2f}%",
                'Baseline Incorrect - with KG': f"{analysis['baseline_incorrect']['with_kg']:.2f}%",
                'Baseline Incorrect - with Both': f"{analysis['baseline_incorrect']['with_both']:.2f}%",
                'Baseline Incorrect - without Paths': f"{analysis['baseline_incorrect']['without_paths']:.2f}%",
                'Baseline Incorrect Total': analysis['baseline_incorrect']['total']
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_excel(output_dir / 'report.xlsx', index=False)

        return analyses
