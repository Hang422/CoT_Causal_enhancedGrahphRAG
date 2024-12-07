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
from src.modules.AccuracyAnalysis import CrossPathAnalyzer
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
            'enhanced': self.cache_root / self.cache_path / 'enhanced',
            'knowledge_graph': self.cache_root / self.cache_path / 'knowledge_graph',
            'causal_graph': self.cache_root / self.cache_path / 'causal_graph',
            'graph_enhanced': self.cache_root / self.cache_path / 'graph_enhanced',
            'llm_enhanced': self.cache_root / self.cache_path / 'llm_enhanced',
            'reasoning': self.cache_root / self.cache_path / 'reasoning',
            'without_enhancement': self.cache_root / self.cache_path / 'without_enhancement',
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
            file_path (str): 数据集标识符（'medinstruct' 或 'medmcqa'）或 Parquet 文件路径
            sample_size (Optional[int]): 可选参数，指定要随机选择的问题数量

        返回：
            List[MedicalQuestion]: 包含问题的列表
        """
        logger = config.get_logger("questions_loader")
        questions = []
        try:
            if file_path == "test2":
                # 加载 medinstruct 数据集
                from datasets import load_dataset
                dataset = load_dataset("cxllin/medinstruct")
                df = pd.DataFrame(dataset['train'])

                if sample_size is not None and sample_size < len(df):
                    df = df.head(sample_size).reset_index(drop=True)

                for idx, row in df.iterrows():
                    try:
                        # 获取整个文本
                        text = row['text']

                        # 移除前缀
                        text = text.replace("Please answer with one of the option in the bracket\nQ:", "").strip()

                        # 分离问题、选项和答案部分
                        parts = text.split("{'A':")
                        if len(parts) != 2:
                            print(f"Warning: Invalid format in row {idx}")
                            continue

                        # 获取问题文本
                        question_text = parts[0].strip().strip('?') + "?"

                        # 处理选项部分
                        options_and_answer = parts[1].strip()
                        options_text = "{" + "'A':" + options_and_answer.split("},")[0] + "}"

                        try:
                            options_dict = eval(options_text)
                        except:
                            print(f"Warning: Could not parse options in row {idx}")
                            continue

                        # 转换选项格式
                        formatted_options = {
                            f'op{chr(97 + i)}': str(value)
                            for i, (key, value) in enumerate(options_dict.items())
                        }

                        # 获取答案
                        answer = options_and_answer.split("},")[1].strip().split(":")[0].strip()
                        answer_idx = ord(answer) - ord('A')
                        formatted_answer = f'op{chr(97 + answer_idx)}'

                        # 创建问题对象
                        question = MedicalQuestion(
                            question=question_text,
                            is_multi_choice=True,
                            options=formatted_options,
                            correct_answer=formatted_answer,
                            topic_name=None
                        )
                        questions.append(question)

                    except Exception as e:
                        print(f"Error processing row {idx}: {str(e)}")
                        continue

            else:
                # medmcqa 部分保持不变
                splits = {'train': 'data/train-00000-of-00001.parquet',
                          'validation': 'data/validation-00000-of-00001.parquet'}

                df = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" + splits["validation"])

                if sample_size is not None and sample_size < len(df):
                    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

                for _, row in df.iterrows():
                    question = MedicalQuestion(
                        question=row['question'],
                        is_multi_choice=True,
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
                        ),
                        topic_name=row['subject_name']
                    )
                    questions.append(question)

        except Exception as e:
            logger.error(f"Questions Initialization Error: {str(e)}")

        if not questions:
            logger.info("Warning: No questions were loaded successfully")
        else:
            logger.info(f"Successfully loaded {len(questions)} questions")

        return questions

    def process_questions(self, questions: List[MedicalQuestion]) -> None:
        """处理问题列表"""
        for i, question in enumerate(questions):
            self.logger.info(f"Processing question {i + 1}/{len(questions)}")
            try:

                cached_question = MedicalQuestion.from_cache(
                    self.cache_paths['original'],
                    question.question
                )

                if not cached_question:
                    question.set_cache_paths(self.cache_paths['original'])
                    question.to_cache()
                else:
                    question = cached_question

                cached_question_derelict = MedicalQuestion.from_cache(
                    self.cache_paths['derelict'],
                    question.question
                )

                if not cached_question_derelict:
                    self.llm.direct_answer(question)
                    question.set_cache_paths(self.cache_paths['derelict'])
                    question.to_cache()
                    question.set_cache_paths(self.cache_paths['knowledge_graph'])
                    question.to_cache()
                    question.set_cache_paths(self.cache_paths['graph_enhanced'])
                    question.to_cache()
                    question.set_cache_paths(self.cache_paths['llm_enhanced'])
                    question.to_cache()
                else:
                    question = cached_question

                cached_question = MedicalQuestion.from_cache(
                    self.cache_paths['causal_graph'],
                    question.question
                )

                if not cached_question:
                    self.processor.generate_initial_causal_graph(question)  # 初步检索
                    # self.llm.causal_only_answer(question)
                    question.set_cache_paths(self.cache_paths['causal_graph'])
                    question.to_cache()
                else:
                    question = cached_question

                # 3. 生成推理链和实体对并缓存
                cached_question = MedicalQuestion.from_cache(
                    self.cache_paths['reasoning'],
                    question.question
                )

                if not cached_question:
                    self.llm.generate_reasoning_chain(question)  # 初步检索-实体对指导检索和裁剪
                    self.processor.process_all_entity_pairs_enhance(question)  # 基于因果裁剪的检索。
                    question.set_cache_paths(self.cache_paths['reasoning'])
                    question.to_cache()
                else:
                    question = cached_question

                # 4. 最终答案并缓存
                cached_question = MedicalQuestion.from_cache(
                    self.cache_paths['without_enhancement'],
                    question.question
                )

                if not cached_question:
                    # self.llm.final_answer_with_all_paths(question)
                    question.set_cache_paths(self.cache_paths['without_enhancement'])
                    question.to_cache()
                else:
                    question = cached_question

                cached_question = MedicalQuestion.from_cache(
                    self.cache_paths['enhanced'],
                    question.question
                )

                if not cached_question:
                    self.llm.enhance_information(question)  # 增强
                    self.llm.answer_with_enhanced_information(question)
                    question.set_cache_paths(self.cache_paths['enhanced'])
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
                        correct_answer=data['correct_answer'],
                        topic_name=data['topic_name'],
                        is_multi_choice=True
                    )
                    questions.append(question)
            except Exception as e:
                print(f"Error loading question from {file}: {str(e)}")
                continue

        self.logger.info(f"Loaded {len(questions)} questions from {original_path}")
        self.process_questions(questions)

    def batch_process_file(self, file_path: str, sample_size: Optional[int] = None) -> None:
        try:
            questions = self.load_questions_from_parquet(file_path, sample_size)
            self.logger.info(f"Loaded {len(questions)} questions from huggingface parquet file")
            self.process_questions(questions)
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")


def main():
    """主函数示例"""
    final1 = 'final_test1_set_3.5'
    final2 = 'final_test1_set_4'
    test = 'test1'
    cache_to_path = 'origin-test2'
    process_path = test
    processor = QuestionProcessor(process_path)

    # processor.batch_process_file('test2', 2000)
    processor.process_from_cache(process_path)

    analyzer = CrossPathAnalyzer(process_path)
    analyzer.analyze_all_stages()


if __name__ == "__main__":
    main()