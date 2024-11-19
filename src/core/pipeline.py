import json

import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

import src.modules.CrossAnalysis
from src.modules.MedicalQuestion import MedicalQuestion
from src.llm.interactor import LLMProcessor
from src.graphrag.entity_processor import EntityProcessor
from src.graphrag.query_processor import QueryProcessor
from config import config
from src.modules.AccuracyAnalysis import QuestionAnalyzer
from src.modules.CrossAnalysis import analyse


class QuestionProcessor:
    """处理医学问题的流水线处理器"""

    def __init__(self, path):
        """初始化处理器"""
        self.llm = LLMProcessor()
        self.logger = config.get_logger("question_processor")

        # 使用config中定义的缓存根目录
        self.cache_root = config.paths["cache"]

        self.cache_path = path

        # 设置不同类型缓存的子目录
        self.cache_paths = {
            'original': self.cache_root / self.cache_path / 'original',
            'derelict': self.cache_root / self.cache_path / 'derelict',
            'casual': self.cache_root / self.cache_path / 'casual',
            'reasoning': self.cache_root / self.cache_path / 'reasoning',
            'final': self.cache_root / self.cache_path / 'final'
        }

        self.processor = QueryProcessor()

        # 创建缓存子目录
        for path in self.cache_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized cache directories under {self.cache_root}")

    @staticmethod
    def load_questions_from_parquet(file_path: str, sample_size: Optional[int] = None) -> List[MedicalQuestion]:
        """从 Parquet 文件加载问题，支持随机抽样。

        参数：
            file_path (str): Parquet 文件的相对路径。
            sample_size (Optional[int]): 可选参数，指定要随机选择的问题数量。

        返回：
            List[MedicalQuestion]: 包含问题的列表。
        """
        questions = []
        data_path = config.paths["data"] / file_path

        splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet',
                  'validation': 'data/validation-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" + splits["validation"])

        try:
            # 读取 Parquet 文件

            # 如果指定了 sample_size，并且小于数据总量，则进行随机抽样
            if sample_size is not None and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

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
            raise Exception(f"加载问题时出错：{data_path}，错误信息：{str(e)}")

        return questions

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
                    self.processor.process_casual_paths(question, 'casual-1')
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
                    self.llm.generate_reasoning_chain(question)
                    self.processor.process_entity_pairs_enhance(question, 'knowledge')
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

    def process_from_cache(self, path: str) -> None:
        """从original缓存目录读取并处理问题"""
        original_path = self.cache_root / path / 'original'

        if not original_path.exists():
            self.logger.error(f"Original cache directory not found: {original_path}")
            return

        questions = []
        for file in original_path.glob('*.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    question = MedicalQuestion(
                        question=data['question'],
                        options=data['options'],
                        correct_answer=data['correct_answer']
                    )
                    questions.append(question)
            except Exception as e:
                print(f"Error loading question from {file}: {str(e)}")
                continue

        self.logger.info(f"Loaded {len(questions)} questions from {original_path}")
        self.process_questions(questions)

    def batch_process_file(self, file_path: str, sample_size: Optional[int] = None, csv=False) -> None:
        try:
            if not csv:
                questions = self.load_questions_from_parquet(file_path, sample_size)
                self.logger.info(f"Loaded {len(questions)} questions from huggingface parquet file")
            else:
                questions = self.load_questions_from_csv(file_path)
                self.logger.info(f"Loaded {len(questions)} questions from {file_path}")
            self.process_questions(questions)
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")


def main():
    """主函数示例"""
    cache_path = '20-gpt-4omini-adaptive-knowledge-0.75-shortest-enhance'
    save_path = f"{cache_path}"
    processor = QuestionProcessor(save_path)
    # 使用相对于data目录的路径
    samples = 'testdata/samples.csv'
    processor.batch_process_file(samples, 100, True)
    #processor.process_from_cache(cache_path)

    analyzer = QuestionAnalyzer(save_path)
    analyzer.save_report(f"{save_path}/report.xlsx")
    analyse(save_path)


if __name__ == "__main__":
    main()
