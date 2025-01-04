import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from src.modules.AccuracyAnalysis import compare_enhanced_with_baseline

def get_paths_from_file(file_path: Path) -> Set[str]:
    """Extract enhanced graph paths from a question file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('enhanced_graph', {}).get('paths', []))
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return set()


def get_all_question_paths(directory: str) -> Dict[str, Set[str]]:
    """Get all question paths from a directory"""
    dir_path = Path(directory)
    question_paths = {}

    # Look through all JSON files in the directory
    for file_path in dir_path.glob('*.json'):
        question_id = file_path.stem  # Use filename without extension as question ID
        paths = get_paths_from_file(file_path)
        question_paths[question_id] = paths

    return question_paths


def compare_enhanced_paths(dir1: str, dir2: str) -> List[str]:
    """
    Compare enhanced graph paths between two directories and identify questions with inconsistent paths.

    Args:
        dir1 (str): Path to first directory
        dir2 (str): Path to second directory

    Returns:
        List[str]: List of question IDs where paths are inconsistent
    """
    # Get paths from both directories
    dir1_paths = get_all_question_paths(dir1)
    dir2_paths = get_all_question_paths(dir2)

    # Find questions with inconsistent paths
    inconsistent_questions = []

    # Check all questions in dir1
    for question_id in dir1_paths:
        if question_id in dir2_paths:
            # Compare paths for the same question
            if dir1_paths[question_id] != dir2_paths[question_id]:
                inconsistent_questions.append(question_id)
                print(f"\nInconsistent paths found for question {question_id}:")
                print(f"Dir1 paths ({len(dir1_paths[question_id])}):")
                for path in sorted(dir1_paths[question_id]):
                    print(f"  {path}")
                print(f"Dir2 paths ({len(dir2_paths[question_id])}):")
                for path in sorted(dir2_paths[question_id]):
                    print(f"  {path}")
        else:
            print(f"Warning: Question {question_id} exists in dir1 but not in dir2")

    # Check for questions in dir2 that aren't in dir1
    for question_id in dir2_paths:
        if question_id not in dir1_paths:
            print(f"Warning: Question {question_id} exists in dir2 but not in dir1")

    # Print summary
    print(f"\nSummary:")
    print(f"Total questions in dir1: {len(dir1_paths)}")
    print(f"Total questions in dir2: {len(dir2_paths)}")
    print(f"Questions with inconsistent paths: {len(inconsistent_questions)}")

    return inconsistent_questions


def compare_model_paths(model1: str, model2: str) -> List[str]:
    """
    Compare enhanced paths between two models' enhanced directories

    Args:
        model1 (str): First model name (e.g., "4o")
        model2 (str): Second model name (e.g., "4")

    Returns:
51be418b6d6e1bbb23919550472a960d.json
96f4707abcabccbce5aa547e0a770672.json
967b542ccdf07e61514bb3dd50a7b723.json
34298524f4a0793a85c134c98a8f6889.json
d20c1f9ad58f50b7a8fc7f1c7b4b61fd.json
        List[str]: List of question IDs with inconsistent paths
    """
    from config import config

    # 构建完整路径
    base_path = config.paths["cache"]
    dir1 = base_path / model1 / "data" / "enhanced"
    dir2 = base_path / model2 / "data" / "enhanced"

    print(f"Comparing enhanced paths between:")
    print(f"Model 1 ({model1}): {dir1}")
    print(f"Model 2 ({model2}): {dir2}")
    print("\n" + "=" * 50 + "\n")

    return compare_enhanced_paths(dir1, dir2)


if __name__ == "__main__":
    # 示例：比较4o和4模型的enhanced paths
    compare_enhanced_with_baseline('temp')  # 对比增强与基线