import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging
from src.modules.MedicalQuestion import MedicalQuestion
from src.llm.interactor import LLMProcessor
from src.graphrag.entity_processor import EntityProcessor
from src.graphrag.query_processor import QueryProcessor
from config import config
from src.modules.AccuracyAnalysis import QuestionAnalyzer

class QuestionProcessor:
    """处理医学问题的流水线处理器"""

    def __init__(self):
        """初始化处理器"""
        self.llm = LLMProcessor()
        self.logger = config.get_logger("question_processor")

        # 使用config中定义的缓存根目录
        self.cache_root = config.paths["cache"]

        # 设置不同类型缓存的子目录
        self.cache_paths = {
            'original': self.cache_root / 'questions' / 'original',
            'derelict': self.cache_root / 'questions' / 'derelict',
            'casual': self.cache_root / 'questions' / 'casual',
            'reasoning': self.cache_root / 'questions' / 'reasoning',
            'final': self.cache_root / 'questions' / 'final'
        }

        self.processor = QueryProcessor()

        # 创建缓存子目录
        for path in self.cache_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized cache directories under {self.cache_root}")

    @staticmethod
    def load_questions_from_csv(file_path: str) -> List[MedicalQuestion]:
        """从CSV文件加载问题"""
        questions = []
        data_path = config.paths["data"] / file_path

        try:
            df = pd.read_csv(data_path)
            for _, row in df.iterrows():
                question = MedicalQuestion(
                    question=row['question'],
                    options={
                        'opa': str(row['opa']),
                        'opb': str(row['opb']),
                        'opc': str(row['opc']),
                        'opd': str(row['opd'])
                    },
                    correct_answer=(
                        'opa' if row['cop'] == 0 else
                        'opb' if row['cop'] == 1 else
                        'opc' if row['cop'] == 2 else
                        'opd' if row['cop'] == 3 else
                        None
                    )
                )
                questions.append(question)
        except Exception as e:
            raise Exception(f"Error loading questions from {data_path}: {str(e)}")

        return questions

    def process_questions(self, questions: List[MedicalQuestion]) -> None:
        """处理问题列表"""
        for i, question in enumerate(questions):
            self.logger.info(f"Processing question {i + 1}/{len(questions)}")

            try:
                # 1. 保存原始问题
                cached_question = MedicalQuestion.from_cache(
                    self.cache_paths['original'],
                    question.question,
                    question.options
                )

                if not cached_question:
                    question.set_cache_paths(self.cache_paths['original'])
                    question.to_cache()
                else:
                    question = cached_question

                cached_question_derelict = MedicalQuestion.from_cache(
                    self.cache_paths['derelict'],
                    question.question,
                    question.options
                )

                if not cached_question_derelict:
                    self.llm.direct_answer(question)
                    question.set_cache_paths(self.cache_paths['derelict'])
                    question.to_cache()
                else:
                    question = cached_question

                cached_question = MedicalQuestion.from_cache(
                    self.cache_paths['casual'],
                    question.question,
                    question.options
                )

                if not cached_question:
                    config.set_database('casual')
                    self.processor.process_casual_paths(question)
                    self.llm.causal_enhanced_answer(question)
                    question.set_cache_paths(self.cache_paths['casual'])
                    question.to_cache()
                else:
                    question = cached_question

                # 3. 生成推理链和实体对并缓存
                cached_question = MedicalQuestion.from_cache(
                    self.cache_paths['reasoning'],
                    question.question,
                    question.options
                )

                if not cached_question:
                    config.set_database('knowledge')
                    self.llm.generate_reasoning_chain(question)
                    self.processor.process_entity_pairs(question)
                    question.set_cache_paths(self.cache_paths['reasoning'])
                    question.to_cache()
                else:
                    question = cached_question

                # 4. 最终答案并缓存
                cached_question = MedicalQuestion.from_cache(
                    self.cache_paths['final'],
                    question.question,
                    question.options
                )

                if not cached_question:
                    self.llm.final_answer_with_all_paths(question)
                    question.set_cache_paths(self.cache_paths['final'])
                    question.to_cache()
                else:
                    question = cached_question

            except Exception as e:
                self.logger.error(f"Error processing question {i + 1}: {str(e)}")
                continue

    def batch_process_file(self, file_path: str) -> None:
        """处理整个CSV文件"""
        try:
            questions = self.load_questions_from_csv(file_path)
            self.logger.info(f"Loaded {len(questions)} questions from {file_path}")
            self.process_questions(questions)
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")


def main():
    """主函数示例"""
    processor = QuestionProcessor()

    # 使用相对于data目录的路径
    file_path = 'testdata/samples.csv'
    processor.batch_process_file(file_path)

    analyzer = QuestionAnalyzer()
    analyzer.save_report()


if __name__ == "__main__":
    main()
