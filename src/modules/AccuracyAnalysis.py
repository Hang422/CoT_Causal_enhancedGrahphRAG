from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd
from src.modules.MedicalQuestion import MedicalQuestion
from config import config
import logging
from src.modules.CrossAnalysis import analyse


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

    def _is_correct(self, question: Dict) -> bool:
        """Check if the answer matches correct_answer"""
        return question.get('answer') == question.get('correct_answer')

    def analyze(self, stage: str) -> Dict:
        """Analyze path presence for baseline correct/incorrect questions"""
        baseline_questions = {q['question']: q for q in self._load_questions('derelict')}
        stage_questions = {q['question']: q for q in self._load_questions(stage)}

        results = {
            "baseline_correct": {
                "with_cg": 0,
                "with_kg": 0,
                "with_both": 0,
                "without_paths": 0,
                "total": 0
            },
            "baseline_incorrect": {
                "with_cg": 0,
                "with_kg": 0,
                "with_both": 0,
                "without_paths": 0,
                "total": 0
            },
            "path_accuracies": {
                "with_cg": {"correct": 0, "total": 0},
                "with_kg": {"correct": 0, "total": 0},
                "with_both": {"correct": 0, "total": 0},
                "without_paths": {"correct": 0, "total": 0}
            }
        }

        # 分析每个问题
        for question in set(baseline_questions.keys()) & set(stage_questions.keys()):
            baseline_q = baseline_questions[question]
            stage_q = stage_questions[question]

            # 确定基线正确/错误
            category = "baseline_correct" if self._is_correct(baseline_q) else "baseline_incorrect"
            results[category]["total"] += 1

            # 检查路径存在情况
            has_causal = bool(stage_q['causal_graph']['paths'] and
                              len(stage_q['causal_graph']['paths']) > 0 and
                              stage_q['causal_graph']['paths'] != "There is no obvious causal relationship.")
            has_knowledge = bool(stage_q['knowledge_graph']['paths'] and
                                 len(stage_q['knowledge_graph']['paths']) > 0)

            # 确定路径类别
            if has_causal and has_knowledge:
                path_category = "with_both"
            elif has_causal:
                path_category = "with_cg"
            elif has_knowledge:
                path_category = "with_kg"
            else:
                path_category = "without_paths"

            # 更新路径类别的总数和正确数
            results["path_accuracies"][path_category]["total"] += 1
            if self._is_correct(stage_q):
                results["path_accuracies"][path_category]["correct"] += 1
                if category == "baseline_correct":
                    results[category][path_category] += 1
                elif category == "baseline_incorrect":
                    results[category][path_category] += 1

        # 转换为百分比
        for category in ["baseline_correct", "baseline_incorrect"]:
            total = results[category]["total"]
            if total > 0:
                for key in ["with_cg", "with_kg", "with_both", "without_paths"]:
                    results[category][key] = (results[category][key] / total) * 100

        # 计算每种路径情况的正确率
        results["path_overall_accuracies"] = {}
        for path_type in ["with_cg", "with_kg", "with_both", "without_paths"]:
            total = results["path_accuracies"][path_type]["total"]
            correct = results["path_accuracies"][path_type]["correct"]
            accuracy = (correct / total * 100) if total > 0 else 0
            results["path_overall_accuracies"][path_type] = accuracy

        # 计算总体正确率
        all_questions = self._load_questions(stage)
        correct_count = sum(1 for q in all_questions if self._is_correct(q))
        overall_accuracy = (correct_count / len(all_questions)) * 100 if all_questions else 0

        results["overall_accuracy"] = overall_accuracy

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
        output_dir = config.paths["output"] / self.cache_root.name
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
