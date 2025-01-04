from pathlib import Path
from typing import Dict, List
import json
import pandas as pd
import logging
from config import config
import json
from pathlib import Path
from typing import Set, List, Dict, Tuple
import shutil
from collections import defaultdict
import hashlib
from config import config
from src.modules.filter import compare_enhanced_with_baseline, filter_by_coverage
logger = config.get_logger("question_processor")

def is_correct(q_data: Dict) -> bool:
    """Check if the answer matches correct_answer"""
    return q_data.get('answer', '').lower() == q_data.get('correct_answer', '').lower()


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

    # 首先使用enhanced来筛选, more strict
    # enhanced_filtered = {q: d for q, d in enhanced_qs.items() if d.get('chain_coverage', {}).get('total_successes', 0)>0}
    enhanced_filtered = {q: d for q, d in enhanced_qs.items() if d.get('enhanced_graph', {}).get('paths', []) and len( d.get('enhanced_graph', {}).get('paths', [])) > 0}
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


def load_questions_with_paths(enhanced_dir: Path) -> Dict[str, dict]:
    questions_with_paths = {}
    if not enhanced_dir.exists():
        print(f"Directory not found: {enhanced_dir}")
        return questions_with_paths

    for file in enhanced_dir.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if (data.get('enhanced_graph', {}).get('paths') and
                    len(data['enhanced_graph']['paths']) > 0):
                questions_with_paths[data['question']] = data
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return questions_with_paths


def find_intersection(model_dirs: List[Path]) -> Set[str]:
    """Find questions common to all directories"""
    questions_by_dir = {}
    for dir_path in model_dirs:
        enhanced_dir = dir_path / 'data' / 'enhanced'
        questions_with_paths = load_questions_with_paths(enhanced_dir)
        questions_by_dir[str(dir_path)] = set(questions_with_paths.keys())
        print(f"Found {len(questions_with_paths)} questions with paths in {dir_path}")

    common_questions = set.intersection(*questions_by_dir.values())
    print(f"\nFound {len(common_questions)} questions common to all directories")
    return common_questions


def copy_filtered_data(src_dir: Path, dest_dir: Path, common_questions: Set[str]) -> None:
    """Copy only intersection questions from source to destination directory"""
    if not src_dir.exists():
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    for file in src_dir.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data['question'] in common_questions:
                new_file = dest_dir / f"{hashlib.md5(data['question'].encode()).hexdigest()}.json"
                with open(new_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error processing file {file}: {e}")


def create_filtered_directories(base_dir: Path, model_dirs: List[str]) -> None:
    """Create filtered versions of directories containing only intersection questions"""
    # Get full paths for model directories
    model_paths = [base_dir / model_dir for model_dir in model_dirs]

    # Find common questions
    common_questions = find_intersection(model_paths)

    # Create filtered versions for each model directory
    stages = ['original','derelict', 'enhanced', 'knowledge_graph', 'remove_llm_enhanced', 'normal_rag', 'remove_enhancer', 'reasoning']

    for model_dir in model_dirs:
        src_base = base_dir / model_dir
        dest_base = base_dir / f"{model_dir}-intersection"

        # Create data directory
        data_dir = dest_base / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)

        # Copy filtered data for each stage
        for stage in stages:
            src_stage = src_base / 'data' / stage
            dest_stage = data_dir / stage

            if src_stage.exists():
                print(f"Processing {model_dir}/{stage}")
                copy_filtered_data(src_stage, dest_stage, common_questions)


def intersect(model_dirs):
    base_dir = Path("/Users/luohang/PycharmProjects/casualGraphRag/cache")
    create_filtered_directories(base_dir, model_dirs)

    STAGES = ['derelict', 'enhanced', 'knowledge_graph', 'remove_llm_enhanced', 'normal_rag', 'remove_enhancer']

    model_dirs_new = [model_dir + '-intersection' for model_dir in model_dirs]
    for dir in model_dirs_new:
        base_dir = dir
        df_report = calculate_accuracies(base_dir, STAGES)
        print(df_report)

        output_path = config.paths["cache"] / base_dir / 'model_accuracy_report.xlsx'
        df_report.to_excel(output_path, index=False)

        compare_enhanced_with_baseline(base_dir)  # 对比增强与基线


if __name__ == "__main__":
    STAGES = ['derelict', 'enhanced',
              'causal_graph', 'knowledge_graph', 'remove_llm_enhanced', 'normal_rag']
    base_dir = '4o-mini-ultra'

    df_report = calculate_accuracies(base_dir, STAGES)
    print(df_report)

    output_path = config.paths["cache"] / base_dir / 'model_accuracy_report.xlsx'
    df_report.to_excel(output_path, index=False)
    print(f"Report saved to {output_path}")

    compare_enhanced_with_baseline(base_dir)  # 对比增强与基线
