from pathlib import Path
from typing import Dict, List
import json
import pandas as pd
import logging
from config import config

logger = config.get_logger("accuracy_calculator")

def is_correct(q_data: Dict) -> bool:
    """Check if the answer matches correct_answer"""
    return q_data.get('answer', '').lower() == q_data.get('correct_answer', '').lower()

def has_enhanced_or_chain_success(q_data: Dict) -> bool:
    """
    Check if question should be included in analysis based on enhanced data:
    Condition: len(enhanced_graph.paths) > 0 or chain_coverage.total_successes > 0
    """
    enhanced_graph = q_data.get('enhanced_graph', {})
    paths = enhanced_graph.get('paths', [])
    chain_cov = q_data.get('chain_coverage', {})
    total_successes = chain_cov.get('total_successes', 0)

    return (isinstance(paths, list) and len(paths) > 0) or (total_successes > 0)

def load_questions_from_dir(dir_path: Path) -> Dict[str, Dict]:
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

def calculate_accuracies(base_dir: str, stages: List[str]) -> pd.DataFrame:
    """
    按照要求：
    1. 使用enhanced筛选：从enhanced中找到有enhanced_graph或chain成功的题目集合(E_set)。
    2. 基线仍为derelict，从derelict中找出与E_set重合的题目集(F_set)。
    3. 在F_set上，以derelict的回答为准，确定baseline_correct_set与baseline_wrong_set。
    4. 对各模型在F_set上的正确率及相对提升率进行统计。

    输出字段：
    - Model
    - Total_Questions: 模型在F_set中的题目数
    - Overall_Accuracy(%): 模型在F_set上的正确率
    - Baseline_Correct_Count, Baseline_Correct_Accuracy(%): 模型在derelict答对子集上的正确率
    - Baseline_Wrong_Count, Baseline_Wrong_Accuracy(%): 模型在derelict答错子集上的正确率
    - Improvement_over_Baseline(%): 相对于derelict整体正确率的提升
    """

    base_path = config.paths["cache"] / base_dir / 'data'
    baseline_stage = "derelict"
    enhanced_stage = "enhanced"

    # 加载所有stage问题
    stage_questions = {}
    for stage in stages:
        stage_dir = base_path / stage
        stage_questions[stage] = load_questions_from_dir(stage_dir)

    # 获取baseline和enhanced数据
    baseline_qs = stage_questions.get(baseline_stage, {})
    enhanced_qs = stage_questions.get(enhanced_stage, {})

    # 首先使用enhanced来筛选
    enhanced_filtered = {q: d for q, d in enhanced_qs.items() if has_enhanced_or_chain_success(d)}

    # 在baseline中存在的最终过滤集合
    filtered_questions = {q: baseline_qs[q] for q in enhanced_filtered if q in baseline_qs}

    if not filtered_questions:
        logger.warning("No questions found after filtering based on enhanced and intersecting with baseline.")
        return pd.DataFrame()

    # 基于derelict（baseline）确定correct/wrong集合
    baseline_total = len(filtered_questions)
    baseline_correct_set = {q for q, d in filtered_questions.items() if is_correct(d)}
    baseline_wrong_set = set(filtered_questions.keys()) - baseline_correct_set

    baseline_correct_num = len(baseline_correct_set)
    baseline_acc = (baseline_correct_num / baseline_total * 100) if baseline_total > 0 else 0.0

    results = []
    for stage in stages:
        q_map = stage_questions.get(stage, {})
        # 只考虑出现在filtered集合中的题目
        considered = {q: q_map[q] for q in filtered_questions.keys() if q in q_map}

        total_count = len(considered)
        if total_count == 0:
            # 此模型在filtered中无数据
            results.append({
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

        # 计算overall accuracy
        total_correct = sum(is_correct(d) for d in considered.values())
        overall_acc = (total_correct / total_count * 100)

        # 在baseline_correct_set子集上的正确率
        bc_questions = {q: considered[q] for q in baseline_correct_set if q in considered}
        bc_count = len(bc_questions)
        bc_correct = sum(is_correct(d) for d in bc_questions.values())
        bc_acc = (bc_correct / bc_count * 100) if bc_count > 0 else 0.0

        # 在baseline_wrong_set子集上的正确率
        bw_questions = {q: considered[q] for q in baseline_wrong_set if q in considered}
        bw_count = len(bw_questions)
        bw_correct = sum(is_correct(d) for d in bw_questions.values())
        bw_acc = (bw_correct / bw_count * 100) if bw_count > 0 else 0.0

        # 相对baseline的提升率
        improvement = overall_acc - baseline_acc

        results.append({
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
    return df

if __name__ == "__main__":
    STAGES = ['derelict', 'enhanced', 'knowledge_graph', 'causal_graph', 'graph_enhanced', 'llm_enhanced']
    base_dir = '1-4omini-4'

    df_report = calculate_accuracies(base_dir, STAGES)
    print(df_report)

    output_path = config.paths["cache"] / base_dir / 'model_accuracy_report.xlsx'
    df_report.to_excel(output_path, index=False)
    print(f"Report saved to {output_path}")