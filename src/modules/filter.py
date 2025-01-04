import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Optional
from src.modules.MedicalQuestion import MedicalQuestion
from config import config
from src.graphrag.query_processor import QueryProcessor


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


def extract_wrong_questions(dir_path: str, t_path: str) -> None:
    """
    从指定目录下提取答错的题目文件（is_correct == False），
    并复制到同级目录中名为 <dir_path>_wrong 的新目录下。

    参数:
        dir_path: 存放题目JSON文件的目录名（在 config.paths["cache"]/<dir_path>/data/ 下）
                  如 "derelict"
    """
    logger = config.get_logger("extract_wrong_questions")
    base_path = Path(config.paths["cache"]) / dir_path / 'data' / t_path
    source_dir = base_path  # 假设你的数据文件存在于 data/derelict 下

    if not source_dir.exists():
        logger.error(f"Directory {source_dir} does not exist.")
        return

    target_dir = base_path / f"{dir_path}_wrong"
    target_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    extracted_count = 0

    for file in source_dir.glob("*.json"):
        q = load_question_from_json(file)
        processed_count += 1
        if q and q.is_correct is False:
            # 答错的题目复制到 target_dir
            shutil.copy2(file, target_dir / file.name)
            extracted_count += 1

    logger.info(f"Processed: {processed_count}")
    logger.info(f"Extracted wrong: {extracted_count}")
    logger.info(f"Extracted files saved to {target_dir}")


def filter_by_enhanced_paths(source_dir: str, target_dir: str) -> None:
    """
    Filter questions that have enhanced graph paths and save their original versions.

    Args:
        source_dir: Source directory path containing the data (e.g., 'test1')
        target_dir: Target directory path to save filtered questions (e.g., 'test1_filtered')
    """
    logger = config.get_logger("path_filter")

    # Set up source paths
    base_source_path = Path(config.paths["cache"]) / source_dir / 'data'
    enhanced_path = base_source_path / 'enhanced'
    original_path = base_source_path / 'original'

    # Set up target path
    target_path = Path(config.paths["cache"]) / target_dir / 'data' / 'original'
    target_path.mkdir(parents=True, exist_ok=True)

    if not all(p.exists() for p in [enhanced_path, original_path]):
        logger.error("Required source directories (enhanced, original) not found")
        return

    processed_count = 0
    retained_count = 0

    logger.info("Filtering questions with enhanced graph paths...")

    for file in enhanced_path.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if enhanced_graph has paths
            enhanced_graph = data.get('enhanced_graph', {})
            paths = enhanced_graph.get('paths', [])

            processed_count += 1

            if paths:  # If there are paths in enhanced_graph
                # Find and copy the corresponding original file
                orig_file = original_path / file.name
                if orig_file.exists():
                    shutil.copy2(orig_file, target_path / file.name)
                    retained_count += 1

        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
            continue

    logger.info(f"Total processed: {processed_count}")
    logger.info(f"Questions with enhanced paths: {retained_count}")
    logger.info(f"Filtered questions saved to {target_path}")

    print("Done filtering by enhanced paths.")


def filter_questions_by_paths(input_path: str, output_path: str) -> None:
    """
    Filter questions based on existence of paths between question and options in the graph.

    Args:
        input_path: Path to directory containing original questions
        output_path: Path to save filtered questions
    """
    # Initialize logger
    logger = config.get_logger("question_processor")

    # Initialize query processor
    query_processor = QueryProcessor()

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    saved_count = 0

    try:
        for file_path in Path(input_path).glob("*.json"):
            try:
                # Load question
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # 确保有必要的字段
                    if 'question' not in data or 'options' not in data:
                        continue

                    # 根据新格式创建问题对象
                    question = MedicalQuestion(
                        question=data['question'],
                        is_multi_choice=True,  # 多选题
                        options=data['options'],
                        correct_answer=data.get('correct_answer', 'opa'),  # 默认值
                        topic_name=data.get('topic_name')
                    )

                processed_count += 1

                # Generate initial causal graph
                query_processor.generate_initial_causal_graph(question)

                # Check if question has valid paths in the graph
                if (hasattr(question.initial_causal_graph, 'paths') and
                        len(question.initial_causal_graph.paths)>0):
                    # Save to output directory
                    output_file = output_dir / file_path.name
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(question.to_dict(), f, indent=2, ensure_ascii=False)
                    saved_count += 1

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue

    finally:
        query_processor.close()

    logger.info(f"Processed {processed_count} questions")
    logger.info(f"Saved {saved_count} questions with valid paths in graph")
    logger.info(f"Filtered out {processed_count - saved_count} questions")


def clean_original_files(directory: str):
   """清理original目录下的json文件,保留所有属性但置空"""
   from pathlib import Path
   import json

   original_dir = config.paths["cache"]/ Path(directory) / 'data' / 'original'
   if not original_dir.exists():
       raise ValueError(f"Original directory not found: {original_dir}")

   for file in original_dir.glob('*.json'):
       with open(file, 'r', encoding='utf-8') as f:
           data = json.load(f)

       clean_data = {
           "question": data["question"],
           "is_multi_choice": data["is_multi_choice"],
           "correct_answer": data["correct_answer"],
           "options": data["options"],
           "topic_name": data["topic_name"],
           "analysis": "",
           "answer": "",
           "confidence": 0.0,
           "normal_results": [],
           "chain_coverage": {},
           "reasoning_chain": [],
           "initial_causal_graph": {"nodes": [], "relationships": [], "paths": []},
           "causal_graph": {"nodes": [], "relationships": [], "paths": []},
           "knowledge_graph": {"nodes": [], "relationships": [], "paths": []},
           "enhanced_graph": {"nodes": [], "relationships": [], "paths": []},
           "enhanced_information": ""
       }

       with open(file, 'w', encoding='utf-8') as f:
           json.dump(clean_data, f, indent=2, ensure_ascii=False)


def cleanup_reasoning_cache(cache_dir: str) -> None:
    """
    Compare enhanced and original directories, then remove corresponding files
    from reasoning directory that are missing in enhanced.

    Args:
        cache_dir (str): Name of the directory under cache (e.g., '40-ultra')
    """
    try:
        # Construct paths
        base_path = config.paths["cache"] / cache_dir / 'data'
        original_path = base_path / 'original'
        enhanced_path = base_path / 'enhanced'
        reasoning_path = base_path / 'reasoning'

        # Verify directories exist
        if not all(p.exists() for p in [original_path, enhanced_path, reasoning_path]):
            raise FileNotFoundError("One or more required directories not found")

        # Get sets of question IDs from original and enhanced
        def get_question_ids(directory: Path) -> set:
            questions = set()
            for file_path in directory.glob('*.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        questions.add(data['question'])
                except (json.JSONDecodeError, KeyError, Exception) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
            return questions

        original_questions = get_question_ids(original_path)
        enhanced_questions = get_question_ids(enhanced_path)

        # Find questions that are in original but not in enhanced
        missing_questions = original_questions - enhanced_questions

        # Delete corresponding files from reasoning directory
        deleted_count = 0
        for file_path in reasoning_path.glob('*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data['question'] in missing_questions:
                        file_path.unlink()
                        deleted_count += 1
            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        print(f"Process completed:")
        print(f"Total questions in original: {len(original_questions)}")
        print(f"Total questions in enhanced: {len(enhanced_questions)}")
        print(f"Questions missing from enhanced: {len(missing_questions)}")
        print(f"Files deleted from reasoning: {deleted_count}")

        # Optional: Create a backup of deleted files
        backup_dir = base_path / 'reasoning_backup'
        if not backup_dir.exists() and deleted_count > 0:
            shutil.copytree(reasoning_path, backup_dir)
            print(f"Backup created at: {backup_dir}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
# Example usage:


if __name__ == "__main__":
    # 示例调用
    # compare_enhanced_with_baseline('1-final-4o-mini')  # 对比增强与基线
    # path = Path('1-final-4o-mini') / 'coverage_filtered'
    # # compare_enhanced_with_baseline(path)
    # filter_by_coverage('1-final-4o-mini', threshold=0.5)  # 覆盖率过滤
    # compare_enhanced_with_baseline(path)
    # extract_wrong_questions('4o-intersection','enhanced')
    # filter_by_enhanced_paths('4o-mini-intersection', '4o-mini-t')
    # path = Path(config.paths["cache"], 'total')
    # output = Path(config.paths["cache"], 'total_filtered')
    # filter_by_enhanced_paths('total_filtered_3.5', 'final-3.5')
    filter_by_enhanced_paths('random', 'final-4o')
    # clean_original_files('4o-mini-ultra')

