from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd
from src.modules.MedicalQuestion import MedicalQuestion
from config import config
import logging
from src.modules.CrossAnalysis import analyse


class UnifiedAnalyzer:
    """Unified analysis system for all types of evaluations"""

    def __init__(self, path):
        self.logger = config.get_logger("unified_analyzer")
        self.cache_root = config.paths["cache"] / path
        self.stages = {
            'derelict': 'Direct Answer',
            'enhanced': 'Enhanced Answer',
            'knowledge_graph': 'Knowledge Graph Only',
            'causal_graph': 'Causal Graph Only',
            'graph_enhanced': 'Graph Enhancement',
            'llm_enhanced': 'LLM Enhancement'
        }

    def get_accuracy(self, stage: str) -> float:
        """Get accuracy for a specific stage"""
        questions = self._load_questions(stage)
        if not questions:
            return 0.0
        return sum(1 for q in questions if q.is_correct) / len(questions)

    def get_basic_analysis(self) -> Dict[str, float]:
        """Get basic accuracy analysis for all stages"""
        return {
            stage_name: self.get_accuracy(stage_key)
            for stage_key, stage_name in self.stages.items()
        }

    def cross_analysis(self, stage1: str, stage2: str) -> Dict:
        """Cross analysis between two stages"""
        questions1 = {q.question: q for q in self._load_questions(stage1)}
        questions2 = {q.question: q for q in self._load_questions(stage2)}

        common_questions = set(questions1.keys()) & set(questions2.keys())

        if not common_questions:
            return {
                'total_questions': 0,
                'comparison': {},
                'details': []
            }

        results = {
            'total_questions': len(common_questions),
            'comparison': {
                'both_correct': 0,
                'both_wrong': 0,
                f'{stage1}_only_correct': 0,
                f'{stage2}_only_correct': 0
            },
            'details': []
        }

        for question in common_questions:
            q1 = questions1[question]
            q2 = questions2[question]

            detail = {
                'question': question,
                'is_multi_choice': q1.is_multi_choice,
                f'{stage1}_answer': q1.answer,
                f'{stage2}_answer': q2.answer,
                'correct_answer': q1.correct_answer,
                f'{stage1}_correct': q1.is_correct,
                f'{stage2}_correct': q2.is_correct
            }
            results['details'].append(detail)

            if q1.is_correct and q2.is_correct:
                results['comparison']['both_correct'] += 1
            elif not q1.is_correct and not q2.is_correct:
                results['comparison']['both_wrong'] += 1
            elif q1.is_correct:
                results['comparison'][f'{stage1}_only_correct'] += 1
            else:
                results['comparison'][f'{stage2}_only_correct'] += 1

        total = len(common_questions)
        results['comparison_percentage'] = {
            key: (value / total * 100) if total > 0 else 0
            for key, value in results['comparison'].items()
        }

        return results

    def get_performance_improvement(self, base_stage: str, compare_stage: str) -> Dict:
        """Analyze performance improvements"""
        analysis = self.cross_analysis(base_stage, compare_stage)
        if analysis['total_questions'] == 0:
            return {'improvement': 0, 'degradation': 0}

        total = analysis['total_questions']
        improvement = analysis['comparison'][f'{compare_stage}_only_correct'] / total * 100
        degradation = analysis['comparison'][f'{base_stage}_only_correct'] / total * 100

        return {
            'improvement': improvement,
            'degradation': degradation,
            'net_change': improvement - degradation
        }

    def generate_report(self, output_path: Optional[str] = None) -> None:
        """Generate complete analysis report"""
        report = {
            'basic_accuracy': self.get_basic_analysis(),
            'cross_analysis': {
                'derelict_vs_enhanced': self.cross_analysis('derelict', 'enhanced'),
                'knowledge_vs_causal': self.cross_analysis('knowledge_graph', 'causal_graph'),
                'graph_vs_llm': self.cross_analysis('graph_enhanced', 'llm_enhanced')
            },
            'improvements': {
                'baseline_to_enhanced': self.get_performance_improvement('derelict', 'enhanced'),
                'knowledge_to_causal': self.get_performance_improvement('knowledge_graph', 'causal_graph'),
                'graph_to_llm': self.get_performance_improvement('graph_enhanced', 'llm_enhanced')
            }
        }

        if output_path is None:
            output_path = self.cache_root / 'analysis_report.json'

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n=== Basic Accuracy Analysis ===")
        for stage_name, accuracy in report['basic_accuracy'].items():
            print(f"{stage_name}: {accuracy:.2%}")

        print("\n=== Performance Improvement Analysis ===")
        for comparison, stats in report['improvements'].items():
            print(f"\n{comparison}:")
            print(f"Improvement: {stats['improvement']:.2%}")
            print(f"Degradation: {stats['degradation']:.2%}")
            print(f"Net Change: {stats['net_change']:.2%}")

    def _load_questions(self, stage: str) -> List[MedicalQuestion]:
        """Load questions from a specific stage"""
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

        return questions


def analyze_experiment(path: str):
    """Convenience function for running analysis"""
    analyzer = UnifiedAnalyzer(path)
    analyzer.generate_report()