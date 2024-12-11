from pathlib import Path
import json
import logging
from typing import Set
from src.modules.MedicalQuestion import MedicalQuestion
from src.graphrag.query_processor import QueryProcessor
from config import config
import shutil


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
                        question.initial_causal_graph.paths != "There is no obvious causal relationship."):
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


def extract_correct_questions(path: str) -> None:
    """找出直接回答正确但enhancement回答错误的问题，并保存原始问题"""
    logger = config.get_logger("correct_extractor")
    cache_root = config.paths["cache"] / path

    # 设置路径
    derelict_path = cache_root / 'derelict'
    enhanced_path = cache_root / 'enhanced'
    original_path = cache_root / 'original'

    # 创建新的目标目录
    target_root = config.paths["cache"] / 'correct' / 'original'
    target_root.mkdir(parents=True, exist_ok=True)

    if not all(p.exists() for p in [derelict_path, enhanced_path, original_path]):
        logger.error("Required directories not found")
        return

    # 读取所有问题
    questions_info = {}
    interesting_questions = []
    enhanced_informations = []

    # 首先读取所有derelict的问题
    logger.info("Loading derelict questions...")
    for file in derelict_path.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 使用新格式创建问题对象
                question = MedicalQuestion(
                    question=data['question'],
                    is_multi_choice=True,
                    options=data['options'],
                    correct_answer=data.get('correct_answer', 'opa'),
                    topic_name=data.get('topic_name'),
                    answer=data.get('answer')
                )
                if question.is_correct:  # 只关注回答正确的问题
                    questions_info[question.question] = question.question
        except Exception as e:
            logger.error(f"Error loading derelict file {file}: {str(e)}")
            continue

    # 然后检查enhancement中这些问题的回答情况
    logger.info("Checking enhancement questions...")
    for file in enhanced_path.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 使用新格式创建问题对象
                question = MedicalQuestion(
                    question=data['question'],
                    is_multi_choice=True,
                    options=data['options'],
                    correct_answer=data.get('correct_answer', 'opa'),
                    topic_name=data.get('topic_name'),
                    answer=data.get('answer'),
                    enhanced_information=data.get('enhanced_information')
                )

                # 如果这个问题在derelict中是正确的，但在enhancement中是错误的
                if question.question in questions_info and not question.is_correct:
                    interesting_questions.append(question.question)
                    enhanced_informations.append(question.enhanced_information)
        except Exception as e:
            logger.error(f"Error loading enhancement file {file}: {str(e)}")
            continue

    # 复制原始问题到新目录
    logger.info("Copying original questions...")
    copied_count = 0
    for file in enhanced_path.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 使用新格式创建问题对象
                question = MedicalQuestion(
                    question=data['question'],
                    is_multi_choice=True,
                    options=data['options'],
                    correct_answer=data.get('correct_answer'),
                    topic_name=data.get('topic_name'),
                    enhanced_information=data.get('enhanced_information'),
                    reasoning_chain=data.get('reasoning_chain')
                )

                if question.question in interesting_questions:
                    # 复制文件到新目录
                    shutil.copy2(file, target_root / file.name)
                    copied_count += 1
        except Exception as e:
            logger.error(f"Error processing original file {file}: {str(e)}")
            continue

    logger.info(f"Found {len(interesting_questions)} questions where derelict was correct but enhancement was wrong")
    logger.info(f"Copied {copied_count} original questions to {target_root}")

    # 打印问题内容供参考
    print(f"\nFound {len(interesting_questions)} questions where direct answer was correct but enhancement was wrong:")
    for idx, question in enumerate(interesting_questions, 1):
        print(f"\n{idx}. {question}")
        print(f"\n{idx}. Enhanced Information:{enhanced_informations[idx - 1]}")


def fiter_causal():
    cache_root = config.paths["cache"]

    # Setup paths
    input_path = cache_root / 'origin-test2' / 'original'
    output_path = cache_root / 'origin-test2-processed' / 'original'

    filter_questions_by_paths(input_path, output_path)


def filter_correct():
    # 示例路径
    path = 'test1'
    extract_correct_questions(path)


if __name__ == "__main__":
    filter_correct()