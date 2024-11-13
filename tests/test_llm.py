import unittest
from dataclasses import replace
from pathlib import Path
from src.modules.MedicalQuestion import MedicalQuestion
from src.llm.interactor import LLMProcessor
from config import config


class TestLLMProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.llm = LLMProcessor()

        # 创建完整的测试问题实例
        cls.test_question = MedicalQuestion(
            # 基本信息
            question="Which of the following is the most likely explanation for decreased nerve conduction velocity in diabetic neuropathy?",
            options={
                "A": "Decreased myelin production",
                "B": "Axonal degeneration",
                "C": "Schwann cell proliferation",
                "D": "Node of Ranvier dysfunction"
            },
            correct_answer="A",

            # 初始化因果关系元素
            casual_nodes=[
                ["Diabetes", "Myelin", "Nerve conduction"],
                ["Hyperglycemia", "Schwann cells", "Myelin"]
            ],
            casual_relationships=[
                ["causes", "reduces"],
                ["impairs", "affects"]
            ],
            casual_paths=[
                "(Diabetes)-[causes]->(Myelin damage)-[reduces]->(Nerve conduction)",
                "(Hyperglycemia)-[impairs]->(Schwann cells)-[affects]->(Myelin)"
            ],

            # 初始化知识元素
            entities_original_pairs={
                'start': ["Diabetes", "Hyperglycemia"],
                'end': ["Nerve conduction", "Myelin"]
            },
            KG_nodes=[
                ["Myelin", "Nerve", "Signal"],
                ["Diabetes", "Peripheral nerves"]
            ],
            KG_relationships=[
                ["component_of", "conducts"],
                ["affects"]
            ],
            KG_paths=[
                "(Myelin)-[component_of]->(Nerve)-[conducts]->(Signal)",
                "(Diabetes)-[affects]->(Peripheral nerves)"
            ]
        )

    def test_direct_answer(self):
        """测试直接回答功能"""
        question = replace(self.test_question)
        try:
            self.llm.direct_answer(question)
            self.assertIsNotNone(question.answer)
            self.assertIsNotNone(question.confidence)
            self.assertIsNotNone(question.reasoning)
        except Exception as e:
            self.fail(f"direct_answer failed with error: {str(e)}")

    def test_causal_enhanced_answer(self):
        """测试因果增强回答功能"""
        question = replace(self.test_question)
        try:
            self.llm.causal_enhanced_answer(question)
            self.assertIsNotNone(question.answer)
            self.assertIsNotNone(question.confidence)
            self.assertIsNotNone(question.reasoning)
        except Exception as e:
            self.fail(f"causal_enhanced_answer failed with error: {str(e)}")

    def test_generate_reasoning_chain(self):
        """测试推理链和实体对生成功能"""
        question = replace(self.test_question)
        try:
            self.llm.generate_reasoning_chain(question)
            self.assertIsNotNone(question.reasoning)
            self.assertIn('start', question.entities_original_pairs)
            self.assertIn('end', question.entities_original_pairs)
            self.assertTrue(len(question.entities_original_pairs['start']) > 0)
            self.assertTrue(len(question.entities_original_pairs['end']) > 0)

            # 打印生成的实体对，用于调试
            print("Generated entity pairs:")
            print(question.entities_original_pairs)
        except Exception as e:
            self.fail(f"generate_reasoning_chain failed with error: {str(e)}")

    def test_final_answer_with_all_paths(self):
        """测试综合所有信息的最终答案生成功能"""
        question = replace(self.test_question)
        try:
            self.llm.final_answer_with_all_paths(question)
            self.assertIsNotNone(question.answer)
            self.assertIsNotNone(question.confidence)
            self.assertIsNotNone(question.reasoning)
        except Exception as e:
            self.fail(f"final_answer_with_all_paths failed with error: {str(e)}")

    def test_consistency_flow(self):
        """测试答案的一致性"""
        try:
            # 直接回答
            direct_question = replace(self.test_question)
            self.llm.direct_answer(direct_question)

            # 因果增强回答
            causal_question = replace(self.test_question)
            self.llm.causal_enhanced_answer(causal_question)

            # 生成推理链
            reasoning_question = replace(self.test_question)
            self.llm.generate_reasoning_chain(reasoning_question)

            # 最终回答
            final_question = replace(self.test_question)
            self.llm.final_answer_with_all_paths(final_question)

            # 验证一致性
            print("\nConsistency check results:")
            print(f"Direct answer: {direct_question.answer}")
            print(f"Causal answer: {causal_question.answer}")
            print(f"Final answer: {final_question.answer}")
            print(f"Correct answer: {self.test_question.correct_answer}")

            if direct_question.answer == self.test_question.correct_answer:
                self.assertEqual(causal_question.answer, self.test_question.correct_answer)
                self.assertEqual(final_question.answer, self.test_question.correct_answer)

        except Exception as e:
            self.fail(f"Consistency test failed with error: {str(e)}")


if __name__ == '__main__':
    unittest.main()