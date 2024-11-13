import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import json
import hashlib
from datetime import datetime
from config import config

from src.modules.MedicalQuestion import (
    Question, QuestionCUIs, GraphPath, QuestionGraphResult,
    EntityPairs, FinalReasoning, QuestionAnalysis, EntityInfo,
    Answer, EntityPairsCUIs, OptionKey
)
from src.graphrag.entity_processor import EntityProcessor
from src.graphrag.query_processor import QueryProcessor
from src.llm.interactor import LLMInteraction


@dataclass
class ProcessingResult:
    """Stores all intermediate and final results for a question"""
    question: Question
    entity_result: Optional[QuestionCUIs] = None
    graph_result: Optional[QuestionGraphResult] = None
    analysis_result: Optional[QuestionAnalysis] = None
    final_reasoning: Optional[FinalReasoning] = None
    answer: Optional[Answer] = None
    error: Optional[str] = None


class BatchProcessor:
    """Handles batch processing of medical questions with caching"""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize batch processor with caching

        Args:
            cache_dir: Directory for caching intermediate results
        """
        self.logger = logging.getLogger(__name__)

        # Setup cache
        self.cache_dir = cache_dir or Path("../../cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create cache subdirectories
        self.entity_cache = self.cache_dir / "entity"
        self.graph_cache = self.cache_dir / "graph"
        self.analysis_cache = self.cache_dir / "analysis"
        self.final_cache = self.cache_dir / "final"

        for directory in [self.entity_cache, self.graph_cache,
                          self.analysis_cache, self.final_cache]:
            directory.mkdir(exist_ok=True)

        # Initialize processors
        self.entity_processor = EntityProcessor()
        self.query_processor = QueryProcessor()
        self.llm_interaction = LLMInteraction()

    def _generate_cache_key(self, data: str) -> str:
        """Generate a unique cache key for input data"""
        return hashlib.md5(data.encode()).hexdigest()

    def _save_to_cache(self, key: str, data: Any, cache_dir: Path) -> None:
        """Save data to cache with timestamp"""
        cache_path = cache_dir / f"{key}.json"
        try:
            # Convert dataclass to dict if necessary
            if hasattr(data, '__dict__'):
                data_dict = asdict(data)
            else:
                data_dict = data

            cache_data = {
                'data': data_dict,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }

            with cache_path.open('w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {str(e)}")

    def _load_from_cache(self, key: str, cache_dir: Path) -> Optional[Any]:
        """Load data from cache if exists"""
        cache_path = cache_dir / f"{key}.json"
        if cache_path.exists():
            try:
                with cache_path.open('r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    return cache_data['data']
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {str(e)}")
        return None

    def process_csv(self, csv_path: Path, output_path: Optional[Path] = None) -> pd.DataFrame:
        """Process questions from CSV file with progress tracking"""
        # Read questions from CSV
        df = pd.read_csv(csv_path)
        self.logger.info(f"Loaded {len(df)} questions from {csv_path}")

        # Convert DataFrame rows to Question objects
        questions = []
        for _, row in df.iterrows():
            try:
                question = Question(
                    text=row['question'],
                    options={
                        OptionKey.A: row['opa'],
                        OptionKey.B: row['opb'],
                        OptionKey.C: row['opc'],
                        OptionKey.D: row['opd'],
                    },
                    correct_answer=(
                        OptionKey.A if row['cop'] == 0 else
                        OptionKey.B if row['cop'] == 1 else
                        OptionKey.C if row['cop'] == 2 else
                        OptionKey.D if row['cop'] == 3 else
                        None
                    )
                )
                questions.append(question)
            except Exception as e:
                self.logger.error(f"Error parsing question: {str(e)}")

        # Process questions
        results = self.process_questions(questions)

        # Convert to DataFrame
        output_df = self._results_to_dataframe(results)

        # Save results
        if output_path:
            output_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved results to {output_path}")

        return output_df

    def process_questions(self, questions: List[Question]) -> List[ProcessingResult]:
        """Process questions with progress bar and caching"""
        results = []

        for question in tqdm(questions, desc="Processing questions"):
            try:
                result = self._process_single_question(question)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing question: {str(e)}")
                results.append(ProcessingResult(
                    question=question,
                    error=str(e)
                ))

        return results

    def _process_single_question(self, question: Question) -> ProcessingResult:
        """Process single question with caching for each step"""
        result = ProcessingResult(question=question)
        question_key = self._generate_cache_key(f"{question.question}{''.join(question.options.values())}")

        try:
            # 1. Entity Processing with cache
            entity_cache = self._load_from_cache(question_key, self.entity_cache)
            if entity_cache:
                result.entity_result = QuestionCUIs(**entity_cache)
            else:
                result.entity_result = self.entity_processor.process_question(question)
                self._save_to_cache(question_key, result.entity_result, self.entity_cache)

            # 2. Graph Query Processing with cache
            config.set_database('casual-1')  # Ensure correct database
            graph_cache = self._load_from_cache(question_key, self.graph_cache)
            if graph_cache:
                result.graph_result = QuestionGraphResult(**graph_cache)
            else:
                result.graph_result = self.query_processor.query_question_cuis(result.entity_result)
                self._save_to_cache(question_key, result.graph_result, self.graph_cache)

            # 3. Initial Analysis with cache
            analysis_cache = self._load_from_cache(question_key, self.analysis_cache)
            if analysis_cache:
                result.analysis_result = QuestionAnalysis(**analysis_cache)
            else:
                result.analysis_result = self.llm_interaction.get_reasoning_chain(
                    question,
                    result.graph_result
                )
                self._save_to_cache(question_key, result.analysis_result, self.analysis_cache)

            # 4. Process entity pairs
            entity_pairs_with_paths = []
            config.set_database('knowledge')  # Switch database
            for pair in result.analysis_result.entity_pairs:
                paths = self.query_processor.query_entity_pairs(pair)
                entity_pairs_with_paths.append((pair, paths))

            # 5. Final Reasoning and Answer
            final_cache = self._load_from_cache(question_key, self.final_cache)
            if final_cache:
                result.answer = Answer(**final_cache)
            else:
                result.final_reasoning = FinalReasoning(
                    question=question,
                    initial_paths=result.graph_result.question_paths,
                    entity_pairs=entity_pairs_with_paths,
                    reasoning_chain=result.analysis_result.reasoning_chain
                )
                result.answer = self.llm_interaction.get_final_answer(result.final_reasoning)
                self._save_to_cache(question_key, result.answer, self.final_cache)

        except Exception as e:
            self.logger.error(f"Error processing question '{question.question[:50]}...': {str(e)}")
            result.error = str(e)

        return result

    def _results_to_dataframe(self, results: List[ProcessingResult]) -> pd.DataFrame:
        """Convert results to DataFrame with error handling"""
        rows = []
        for result in results:
            try:
                row = {
                    'question': result.question.question,
                    'answer': result.answer.answer.value if result.answer else None,
                    'confidence': result.answer.confidence if result.answer else None,
                    'explanation': result.answer.explanation if result.answer else None,
                    'correct_answer': result.question.correct_answer.value if result.question.correct_answer else None,
                    'is_correct': result.answer.isCorrect if result.answer else None,
                    'error': result.error
                }
                rows.append(row)
            except Exception as e:
                self.logger.error(f"Error converting result to row: {str(e)}")
                rows.append({
                    'question': result.question.question if result.question else 'Unknown',
                    'error': str(e)
                })

        return pd.DataFrame(rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        self.query_processor.close()


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize processor with default cache directory
    processor = BatchProcessor()

    # Process CSV file
    results_df = processor.process_csv(
        csv_path=Path("../../data/testdata/test_sample.csv"),
        output_path=Path("../../data/results/results.csv")
    )