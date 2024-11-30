from pathlib import Path
from typing import Dict, List
import json
import shutil
from src.modules.MedicalQuestion import MedicalQuestion
from config import config
import logging


def extract_correct_questions(path: str) -> None:
    """找出直接回答正确但enhancement回答错误的问题，并保存原始问题"""
    logger = config.get_logger("correct_extractor")
    cache_root = config.paths["cache"] / path

    # 设置路径
    derelict_path = cache_root / 'derelict'
    enhancement_path = cache_root / 'enhancement'
    original_path = cache_root / 'original'

    # 创建新的目标目录
    target_root = config.paths["cache"] / 'correct' / 'original'
    target_root.mkdir(parents=True, exist_ok=True)

    if not all(p.exists() for p in [derelict_path, enhancement_path, original_path]):
        logger.error("Required directories not found")
        return

    # 读取所有问题
    questions_info = {}
    interesting_questions = []

    # 首先读取所有derelict的问题
    logger.info("Loading derelict questions...")
    for file in derelict_path.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                question = MedicalQuestion(**data)
                if question.is_correct:  # 只关注回答正确的问题
                    questions_info[question.question] = question.question
        except Exception as e:
            logger.error(f"Error loading derelict file {file}: {str(e)}")
            continue

    # 然后检查enhancement中这些问题的回答情况
    logger.info("Checking enhancement questions...")
    for file in enhancement_path.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                question = MedicalQuestion(**data)
                # 如果这个问题在derelict中是正确的，但在enhancement中是错误的
                if question.question in questions_info and not question.is_correct:
                    interesting_questions.append(question.question)
        except Exception as e:
            logger.error(f"Error loading enhancement file {file}: {str(e)}")
            continue

    # 复制原始问题到新目录
    logger.info("Copying original questions...")
    copied_count = 0
    for file in original_path.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                question = MedicalQuestion(**data)
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


def main():
    # 示例路径
    path = 'test'
    extract_correct_questions(path)


if __name__ == "__main__":
    main()