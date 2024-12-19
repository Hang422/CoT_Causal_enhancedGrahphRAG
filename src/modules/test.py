from pathlib import Path
from typing import Dict, List, Set
import json
import pandas as pd
import logging
from config import config

logger = config.get_logger("topic_accuracy_calculator")


def is_correct(q_data: Dict) -> bool:
    """Check if the answer matches correct_answer"""
    return q_data.get('answer', '').lower() == q_data.get('correct_answer', '').lower()


def load_questions_from_dir(dir_path: Path) -> Dict[str, Dict]:
    """Load all questions from a directory"""
    questions = {}
    if not dir_path.exists():
        logger.warning(f"Directory not found: {dir_path}")
        return questions

    for f in dir_path.glob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                q_data = json.load(fh)
            if 'question' in q_data and q_data['question']:
                questions[q_data['question']] = q_data
        except Exception as e:
            logger.error(f"Error loading {f}: {e}")
    return questions


def calculate_topic_accuracies(base_dir: str, stages: List[str]) -> pd.DataFrame:
    """
    Calculate accuracies by topic for each model stage.

    Args:
        base_dir: Base directory containing the data
        stages: List of model stages to evaluate

    Returns:
        DataFrame with accuracy metrics by topic for each model
    """
    base_path = config.paths["cache"] / base_dir / 'data'
    baseline_stage = "derelict"
    enhanced_stage = "enhanced"

    # Load questions for all stages
    stage_questions = {}
    for stage in stages:
        stage_dir = base_path / stage
        stage_questions[stage] = load_questions_from_dir(stage_dir)

    baseline_qs = stage_questions.get(baseline_stage, {})
    enhanced_qs = stage_questions.get(enhanced_stage, {})

    # Filter questions that have enhanced paths
    enhanced_filtered = {q: d for q, d in enhanced_qs.items()
                         if d.get('chain_coverage', {}).get('total_successes', 0) > 0}

    # Get questions that exist in baseline
    filtered_questions = {q: baseline_qs[q] for q in enhanced_filtered if q in baseline_qs}

    if not filtered_questions:
        logger.warning("No questions found after filtering")
        return pd.DataFrame()

    # Group questions by topic
    topic_questions: Dict[str, Dict[str, Dict]] = {}
    for q, data in filtered_questions.items():
        topic = data.get('topic_name', 'Unknown')
        if topic not in topic_questions:
            topic_questions[topic] = {}
        topic_questions[topic][q] = data

    results = []
    for topic, topic_qs in topic_questions.items():
        # Calculate baseline metrics for this topic
        topic_total = len(topic_qs)
        baseline_correct_set = {q for q, d in topic_qs.items() if is_correct(d)}
        baseline_wrong_set = set(topic_qs.keys()) - baseline_correct_set

        baseline_correct_num = len(baseline_correct_set)
        baseline_acc = (baseline_correct_num / topic_total * 100) if topic_total > 0 else 0.0

        # Calculate metrics for each stage
        for stage in stages:
            q_map = stage_questions.get(stage, {})
            # Consider only questions in the filtered set for this topic
            considered = {q: q_map[q] for q in topic_qs.keys() if q in q_map}

            total_count = len(considered)
            if total_count == 0:
                results.append({
                    'Topic': topic,
                    'Model': stage,
                    'Total_Questions': 0,
                    'Overall_Accuracy(%)': 0.0,
                    'Baseline_Correct_Count': len(baseline_correct_set),
                    'Baseline_Correct_Accuracy(%)': 0.0,
                    'Baseline_Wrong_Count': len(baseline_wrong_set),
                    'Baseline_Wrong_Accuracy(%)': 0.0,
                    'Improvement_over_Baseline(%)': 0.0
                })
                continue

            # Calculate overall accuracy
            total_correct = sum(is_correct(d) for d in considered.values())
            overall_acc = (total_correct / total_count * 100)

            # Calculate accuracy on baseline correct set
            bc_questions = {q: considered[q] for q in baseline_correct_set if q in considered}
            bc_count = len(bc_questions)
            bc_correct = sum(is_correct(d) for d in bc_questions.values())
            bc_acc = (bc_correct / bc_count * 100) if bc_count > 0 else 0.0

            # Calculate accuracy on baseline wrong set
            bw_questions = {q: considered[q] for q in baseline_wrong_set if q in considered}
            bw_count = len(bw_questions)
            bw_correct = sum(is_correct(d) for d in bw_questions.values())
            bw_acc = (bw_correct / bw_count * 100) if bw_count > 0 else 0.0

            # Calculate improvement over baseline
            improvement = overall_acc - baseline_acc

            results.append({
                'Topic': topic,
                'Model': stage,
                'Total_Questions': total_count,
                'Overall_Accuracy(%)': overall_acc,
                'Baseline_Correct_Count': bc_count,
                'Baseline_Correct_Accuracy(%)': bc_acc,
                'Baseline_Wrong_Count': bw_count,
                'Baseline_Wrong_Accuracy(%)': bw_acc,
                'Improvement_over_Baseline(%)': improvement
            })

    df = pd.DataFrame(results)

    # Sort by Topic and Model
    df = df.sort_values(['Topic', 'Model'])

    return df


def save_topic_accuracies(base_dir: str, stages: List[str]) -> None:
    """
    Calculate and save topic-based accuracies to Excel files.

    Args:
        base_dir: Base directory containing the data
        stages: List of model stages to evaluate
    """
    df_report = calculate_topic_accuracies(base_dir, stages)

    if df_report.empty:
        logger.warning("No results to save")
        return

    # Save full report
    output_path = config.paths["cache"] / base_dir / 'topic_accuracy_report.xlsx'
    df_report.to_excel(output_path, index=False)
    logger.info(f"Full report saved to {output_path}")

    # Create summary by topic
    summary_path = config.paths["cache"] / base_dir / 'topic_accuracy_summary.xlsx'
    topic_summary = df_report.pivot_table(
        index='Topic',
        columns='Model',
        values=['Overall_Accuracy(%)', 'Improvement_over_Baseline(%)'],
        aggfunc='mean'
    ).round(2)

    topic_summary.to_excel(summary_path)
    logger.info(f"Topic summary saved to {summary_path}")


STAGES = ['derelict', 'enhanced', 'enhanced_without_initial', 'causal_graph',
          'knowledge_graph', 'remove_llm_enhanced', 'normal_rag']
base_dir = 'origin-1'  # or whatever your base directory is

save_topic_accuracies(base_dir, STAGES)