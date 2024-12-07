from pathlib import Path
import json
import logging
from typing import Set
from src.modules.MedicalQuestion import MedicalQuestion
from src.graphrag.query_processor import QueryProcessor
from config import config


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


    # Process all questions in input directory
    processed_count = 0
    saved_count = 0

    try:
        for file_path in Path(input_path).glob("*.json"):
            try:
                # Load question
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    question = MedicalQuestion(**data)

                processed_count += 1

                query_processor.generate_initial_causal_graph(question)
                # Check if question has valid paths in the graph
                if question.initial_causal_graph.paths != "There is no obvious causal relationship.":
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


if __name__ == "__main__":
    cache_root = config.paths["cache"]

    # Setup paths
    input_path = cache_root / 'origin-test2' / 'original'
    output_path = cache_root / 'origin-test2-processed' / 'original'

    filter_questions_by_paths(input_path, output_path)