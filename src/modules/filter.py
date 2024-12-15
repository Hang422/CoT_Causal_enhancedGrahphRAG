import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Optional
from src.modules.MedicalQuestion import MedicalQuestion
from config import config


def load_question_from_json(file_path: Path) -> Optional[MedicalQuestion]:
    """从JSON文件加载Question对象"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        question = MedicalQuestion(
            question=data.get('question'),
            is_multi_choice=data.get('is_multi_choice', True),
            options=data.get('options', {}),
            correct_answer=data.get('correct_answer', 'opa'),
            topic_name=data.get('topic_name'),
            answer=data.get('answer'),
            reasoning_chain=data.get('reasoning_chain'),
            enhanced_information=data.get('enhanced_information'),
            initial_causal_graph=data.get('initial_causal_graph'),
            causal_graph=data.get('causal_graph'),
            knowledge_graph=data.get('knowledge_graph'),
            enhanced_graph=data.get('enhanced_graph'),
            chain_coverage=data.get('chain_coverage')
        )
        return question
    except Exception as e:
        logger = config.get_logger("load_question")
        logger.error(f"Error loading {file_path}: {e}")
        return None


def compare_enhanced_with_baseline(dir_path: str) -> None:
    """
    对比增强(enhanced)与基线(derelict)的回答正确性，并在dir_path下输出四个子目录：
    1. base_correct_enhanced_wrong：基线对但增强错的问题
    2. base_wrong_enhanced_correct：基线错但增强对的问题
    注：这两个目录中保存的是original下对应的原始文件副本，方便后续分析

    Args:
        dir_path: 包含 original、derelict、enhanced 三个子目录的路径
    """
    logger = config.get_logger("compare_enhanced_with_baseline")
    base_path = Path(config.paths["cache"]) / dir_path / 'data'

    derelict_path = base_path / 'derelict'
    enhanced_path = base_path / 'enhanced'
    original_path = base_path / 'original'

    if not all(p.exists() for p in [derelict_path, enhanced_path, original_path]):
        logger.error("Required directories (derelict, enhanced, original) not found under given dir_path.")
        return

    # 创建输出目录
    base_correct_enhanced_wrong_dir = Path(config.paths["cache"]) / dir_path / 'base_correct_enhanced_wrong'
    base_wrong_enhanced_correct_dir = Path(config.paths["cache"]) / dir_path / 'base_wrong_enhanced_correct'

    base_correct_enhanced_wrong_dir.mkdir(parents=True, exist_ok=True)
    base_wrong_enhanced_correct_dir.mkdir(parents=True, exist_ok=True)

    # 读取基线答案情况
    base_correctness = {}
    for file in derelict_path.glob("*.json"):
        q = load_question_from_json(file)
        if q and q.question:
            base_correctness[q.question] = q.is_correct

    # 对比enhanced与baseline
    base_correct_enhanced_wrong_count = 0
    base_wrong_enhanced_correct_count = 0

    for file in enhanced_path.glob("*.json"):
        q = load_question_from_json(file)
        if q.chain_coverage.get('total_successes') == 0:
            continue
        if not q or not q.question:
            continue
        base_correct = base_correctness.get(q.question, False)
        enhanced_correct = q.is_correct

        # 基线对增强错
        if base_correct and not enhanced_correct:
            # 找到original文件并复制
            orig_file = enhanced_path / file.name
            if orig_file.exists():
                shutil.copy2(orig_file, base_correct_enhanced_wrong_dir / file.name)
                base_correct_enhanced_wrong_count += 1

        # 基线错增强对
        if (not base_correct) and enhanced_correct:
            orig_file = enhanced_path / file.name
            if orig_file.exists():
                shutil.copy2(orig_file, base_wrong_enhanced_correct_dir / file.name)
                base_wrong_enhanced_correct_count += 1

    logger.info(f"基线对增强错数量: {base_correct_enhanced_wrong_count}")
    logger.info(f"基线错增强对数量: {base_wrong_enhanced_correct_count}")

    print("Done comparing enhanced with baseline.")


def filter_by_coverage(dir_path: str, threshold: float = 0.5) -> None:
    """
    根据 coverage_threshold 对问题进行过滤。
    只保留那些 enhanced 下 chain_coverage 中 success_counts 非零比例高于 threshold 的问题，
    并将对应的 enhanced、derelict、original 文件复制到新的子目录 coverage_filtered 下。

    Args:
        dir_path: 包含 original、derelict、enhanced 三个子目录的路径
        threshold: 成功步骤非零比例的阈值 (0.5表示至少一半的success_counts>0)
    """
    logger = config.get_logger("coverage_filter")
    base_path = Path(config.paths["cache"]) / dir_path / 'data'

    enhanced_path = base_path / 'enhanced'
    derelict_path = base_path / 'derelict'
    original_path = base_path / 'original'

    if not all(p.exists() for p in [enhanced_path, derelict_path, original_path]):
        logger.error("Required directories not found")
        return

    target_root = Path(config.paths["cache"]) / dir_path / 'coverage_filtered' / 'data'
    target_enhanced = target_root / 'enhanced'
    target_derelict = target_root / 'derelict'
    target_original = target_root / 'original'

    for d in [target_enhanced, target_derelict, target_original]:
        d.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    retained_count = 0

    logger.info("Filtering by coverage...")

    for file in enhanced_path.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            chain_coverage = data.get('chain_coverage', {})
            success_counts = chain_coverage.get('success_counts', [])

            if success_counts:
                non_zero_count = sum(1 for c in success_counts if c > 0)
                coverage_ratio = non_zero_count / len(success_counts)
            else:
                coverage_ratio = 0

            processed_count += 1

            if coverage_ratio >= threshold:
                # 复制enhanced文件
                shutil.copy2(file, target_enhanced / file.name)

                # derelict文件
                der_file = derelict_path / file.name
                if der_file.exists():
                    shutil.copy2(der_file, target_derelict / file.name)

                # original文件
                orig_file = original_path / file.name
                if orig_file.exists():
                    shutil.copy2(orig_file, target_original / file.name)

                retained_count += 1

        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
            continue

    logger.info(f"Processed: {processed_count}")
    logger.info(f"Retained: {retained_count} (coverage >= {threshold})")
    logger.info(f"Result saved to {target_root}")

    print("Done filtering by coverage.")


if __name__ == "__main__":
    # 示例调用
    compare_enhanced_with_baseline('1-4omini-4')  # 对比增强与基线
    path = Path('1-4omini-4') / 'coverage_filtered'
    compare_enhanced_with_baseline(path)
    filter_by_coverage(path, threshold=0.5)  # 覆盖率过滤
