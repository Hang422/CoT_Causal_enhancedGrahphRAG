import gc
import hashlib
import json
from typing import List, Optional
import pandas as pd
from config import config
from src.graphrag.graph_enhancer import EnhancedGraphEnhancer
from src.graphrag.query_processor import QueryProcessor
from src.llm.interactor import LLMProcessor
from src.modules.AccuracyAnalysis import calculate_accuracies
from src.modules.MedicalQuestion import MedicalQuestion, SubGraph
from src.modules.filter import compare_enhanced_with_baseline
from src.modules.AccuracyAnalysis import intersect

class QuestionProcessor:
    """处理医学问题的流水线处理器"""

    def __init__(self, path):
        """初始化处理器"""
        self.llm = LLMProcessor(path)
        self.logger = config.get_logger("question_processor")

        # 使用config中定义的缓存根目录
        self.cache_root = config.paths["cache"]

        self.cache_path = path

        # 设置不同类型缓存的子目录
        self.cache_paths = {
            'original': self.cache_root / self.cache_path / 'data' / 'original',
            'derelict': self.cache_root / self.cache_path / 'data' / 'derelict',
            'enhanced': self.cache_root / self.cache_path / 'data' / 'enhanced',
            'reasoning': self.cache_root / self.cache_path / 'data' / 'reasoning',
            'knowledge_graph': self.cache_root / self.cache_path / 'data' / 'knowledge_graph',
            'causal_graph': self.cache_root / self.cache_path / 'data' / 'causal_graph',
            'remove_llm_enhanced': self.cache_root / self.cache_path / 'data' / 'remove_llm_enhanced',
            'remove_enhancer': self.cache_root / self.cache_path / 'data' / 'remove_enhancer',
            'enhanced_without_chain': self.cache_root / self.cache_path / 'data' / 'enhanced_without_chain',
            'normal_rag': self.cache_root / self.cache_path / 'data' / 'normal_rag'
        }

        self.processor = QueryProcessor()
        self.enhancer = EnhancedGraphEnhancer()

        # 创建缓存子目录
        for path in self.cache_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized cache directories under {self.cache_root}")

    def complete_process_question(self, question: MedicalQuestion, initial: bool):
        # self.processor.generate_initial_causal_graph(question)
        cached_question = MedicalQuestion.from_cache(
            self.cache_paths['enhanced'],
            question.question
        )
        if not cached_question:
            self.processor.process_chain_of_thoughts(question, 'both', True)
            self.enhancer.enhance_graphs(question)
            if len(question.enhanced_graph.paths) == 0:
                self.logger.warning(f"Question{hashlib.md5(question.question.encode()).hexdigest()} has no enhanced graphs")
                return False
            self.llm.enhance_information_with_chain(question, '_complete_with_chain')  # 增强
            self.llm.answer_with_enhanced_information(question, '_complete_with_chain')
            question.set_cache_paths(self.cache_paths['enhanced'])
            question.to_cache()
            return True

    def ablation_remove_enhancer(self, question: MedicalQuestion):
        cached_question = MedicalQuestion.from_cache(
            self.cache_paths['remove_enhancer'],
            question.question
        )
        if not cached_question:
            self.processor.process_chain_of_thoughts(question, 'both', True)
            question.enhanced_graph.clear()
            question.enhanced_graph.paths.extend(question.causal_graph.paths)
            question.enhanced_graph.paths.extend(question.knowledge_graph.paths)
            if len(question.enhanced_graph.paths) == 0:
                self.logger.warning(f"Question{hashlib.md5(question.question.encode()).hexdigest()} has no enhanced graphs")
                return False
            self.llm.enhance_information_with_chain(question, '_without_enhancer')  # 增强
            self.llm.answer_with_enhanced_information(question, '_without_enhancer')
            question.set_cache_paths(self.cache_paths['remove_enhancer'])
            question.to_cache()
            return True

    def complete_process_question_without_chain(self, question: MedicalQuestion, initial: bool):
        # self.processor.generate_initial_causal_graph(question)
        cached_question = MedicalQuestion.from_cache(
            self.cache_paths['enhanced_without_chain'],
            question.question
        )
        if not cached_question:
            self.processor.process_chain_of_thoughts(question, 'both', True)
            self.enhancer.enhance_graphs(question)
            if len(question.enhanced_graph.paths) != 0:
                self.llm.enhance_information(question, '_complete')  # 增强
                self.llm.answer_with_enhanced_information(question, '_complete')
            question.set_cache_paths(self.cache_paths['enhanced_without_chain'])
            question.to_cache()

    def ablation_using_causal_only(self, question: MedicalQuestion):
        # self.processor.generate_initial_causal_graph(question)
        cached_question = MedicalQuestion.from_cache(
            self.cache_paths['causal_graph'],
            question.question
        )
        if not cached_question:
            self.processor.process_chain_of_thoughts(question, 'causal', True)
            self.enhancer.enhance_graphs(question)
            if len(question.enhanced_graph.paths) == 0:
                self.logger.warning(f"Question{question.question} has no causal graphs")
                return False
            self.llm.enhance_information(question, '_causal_only')  # 增强
            self.llm.answer_with_enhanced_information(question, '_causal_only')
            question.set_cache_paths(self.cache_paths['causal_graph'])
            question.to_cache()

    def ablation_using_knowledge_only(self, question: MedicalQuestion):
        cached_question = MedicalQuestion.from_cache(
            self.cache_paths['knowledge_graph'],
            question.question
        )
        if not cached_question:
            question.initial_causal_graph.paths = []
            self.processor.process_chain_of_thoughts(question, 'knowledge', False)
            self.enhancer.enhance_graphs(question)
            self.llm.enhance_information_with_chain(question, '_kg_only')  # 增强
            self.llm.answer_with_enhanced_information(question, '_kg_only')
            question.set_cache_paths(self.cache_paths['knowledge_graph'])
            question.to_cache()

    def ablation_llm_enhanced(self, question: MedicalQuestion):
        # self.processor.generate_initial_causal_graph(question)
        cached_question = MedicalQuestion.from_cache(
            self.cache_paths['remove_llm_enhanced'],
            question.question
        )
        if not cached_question:
            self.processor.process_chain_of_thoughts(question, 'both', True)
            self.enhancer.enhance_graphs(question)
            self.llm.answer_with_cot(question)
            question.set_cache_paths(self.cache_paths['remove_llm_enhanced'])
            question.to_cache()

    def normal_rag(self, question: MedicalQuestion):
        cached_question = MedicalQuestion.from_cache(
            self.cache_paths['normal_rag'],
            question.question
        )
        if not cached_question:
            self.processor.process_vector_search(question)
            self.llm.answer_normal_rag(question)
            question.set_cache_paths(self.cache_paths['normal_rag'])
            question.to_cache()

    def compare_experiments(self, question: MedicalQuestion):
        self.ablation_remove_enhancer(question)
        self.ablation_llm_enhanced(question)
        self.ablation_using_knowledge_only(question)
        self.normal_rag(question)

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
                #
                # cached_question = MedicalQuestion.from_cache(
                #     self.cache_paths['derelict'],
                #     question.question
                # )
                # if not cached_question:
                #     self.llm.direct_answer(question)
                #     question.set_cache_paths(self.cache_paths['derelict'])
                #     question.to_cache()
                # else:
                #     question = cached_question

                cached_question = MedicalQuestion.from_cache(
                    self.cache_paths['reasoning'],
                    question.question
                )
                if not cached_question:
                    self.llm.generate_reasoning_chain(question)
                    question.set_cache_paths(self.cache_paths['reasoning'])
                    question.to_cache()
                else:
                    question = cached_question

                if not self.complete_process_question(question, False):
                    continue

                self.compare_experiments(question)

            except Exception as e:
                self.logger.error(f"Error processing question {i + 1}: {str(e)}")
                continue

    def process_from_cache(self, path: str) -> None:
        """从original缓存目录读取并处理问题"""
        original_path = self.cache_root / path / 'data' / 'original'

        if not original_path.exists():
            self.logger.error(f"Original cache directory not found: {original_path}")
            return

        questions = []
        loaded_count = 0
        error_count = 0

        for file in original_path.glob('*.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # 处理 SubGraph 结构
                    # initial_causal_graph = SubGraph.from_dict(data.get('initial_causal_graph', {}))

                    # 创建 MedicalQuestion 实例，包含所有字段
                    question = MedicalQuestion(
                        question=data['question'],
                        is_multi_choice=data['is_multi_choice'],
                        correct_answer=data['correct_answer'],
                        options=data['options'],
                        topic_name=data['topic_name'],
                        # initial_causal_graph=initial_causal_graph,
                        reasoning_chain=data['reasoning_chain']
                    )

                    questions.append(question)
                    loaded_count += 1

            except Exception as e:
                self.logger.error(f"Error loading question from {file}: {str(e)}")
                error_count += 1
                continue

        self.logger.info(f"Successfully loaded {loaded_count} questions")
        if error_count > 0:
            self.logger.warning(f"Encountered {error_count} errors while loading questions")

        if questions:
            self.process_questions(questions)

    def batch_process_file(self, file_path: str, sample_size: Optional[int] = None) -> None:
        try:
            questions = self.load_questions_from_parquet(file_path, sample_size)
            self.logger.info(f"Loaded {len(questions)} questions from huggingface parquet file")
            self.process_questions(questions)
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")

    @staticmethod
    def load_questions_from_parquet(file_path: str, sample_size: Optional[int] = None) -> List[MedicalQuestion]:
        """从 Parquet 文件加载问题，仅选择生理学、生物化学和药理学领域的题目。

        参数：
            file_path (str): 数据集标识符（'medinstruct' 或 'medmcqa'）或 Parquet 文件路径
            sample_size (Optional[int]): 可选参数，指定要随机选择的问题数量

        返回：
            List[MedicalQuestion]: 包含问题的列表
        """
        logger = config.get_logger("questions_loader")
        questions = []
        # 定义目标学科
        target_subjects = {'Physiology', 'Biochemistry', 'Pharmacology'}
        try:
            if file_path == "test2":
                # 加载 medinstruct 数据集
                from datasets import load_dataset
                dataset = load_dataset("cxllin/medinstruct")
                df = pd.DataFrame(dataset['train'])

                # Note: medinstruct dataset doesn't have subject categorization
                if sample_size is not None and sample_size < len(df):
                    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

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
                # medmcqa 部分
                splits = {'train': 'data/train-00000-of-00001.parquet',
                          'validation': 'data/validation-00000-of-00001.parquet'}

                df = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" + splits["train"])

                # 只保留目标学科的问题
                df = df[df['subject_name'].isin(target_subjects)].reset_index(drop=True)

                if sample_size is not None and sample_size < len(df):
                    df = df.head(sample_size).reset_index(drop=True)  # 将 sample 改为 head

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

            if not questions:
                logger.info("Warning: No questions were loaded successfully")
            else:
                logger.info(f"Successfully loaded {len(questions)} questions from {', '.join(target_subjects)}")

        except Exception as e:
            logger.error(f"Questions Initialization Error: {str(e)}")

        return questions


def compare_models(process_path):
    """主函数示例"""
    try:
        processor = QuestionProcessor(process_path)
        processor.process_from_cache(process_path)

        STAGES = ['derelict', 'enhanced', 'knowledge_graph', 'remove_llm_enhanced', 'normal_rag', 'remove_enhancer']
        base_dir = process_path
        df_report = calculate_accuracies(base_dir, STAGES)
        print(df_report)

        output_path = config.paths["output"] / f'{process_path}.xlsx'
        df_report.to_excel(output_path, index=False)

        compare_enhanced_with_baseline(process_path)  # 对比增强与基线


    finally:
        # 手动清理内存
        print("开始清理内存...")
        # del processor, df_report # 删除变量引用
        gc.collect()  # 执行垃圾回收
        print("内存清理完成！")


if __name__ == "__main__":
    # models = ['4', '4o', '4o-mini']
    # intersect(models)
    config.openai['model'] = 'gpt-4o-mini'
    compare_models('4o-mini-intersection')
    config.openai['model'] = 'gpt-4o'
    compare_models('4o-intersection')
    config.openai['model'] = 'gpt-4-turbo'
    compare_models('4-intersection')
    compare_models('2-mini_1-4o')
    compare_models('chain=40-rest=mini')
    compare_models('chain=mini-rest=4o')
