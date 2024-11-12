import unittest

import pandas as pd
from src.llm.interactor import LLMInteraction
from src.modules.data_format import Question, OptionKey
from tests.example import question, question_graph_result, final_reasoning
from src.graphrag.entity_processor import EntityProcessor
from src.graphrag.query_processor import QueryProcessor
from config import config


def test_llm_direct_answer():
    llm = LLMInteraction()
    answer = llm.get_direct_answer(question)
    assert answer.answer == question.correct_answer


def test_llm_casual_answer():
    llm = LLMInteraction()
    answer_direct = llm.get_direct_answer(question)
    answer_casual = llm.get_causal_enhanced_answer(question, question_graph_result)
    if answer_direct.answer == question.correct_answer:
        assert answer_casual.answer == question.correct_answer


def test_llm_get_chains():
    llm = LLMInteraction()
    answer = llm.get_reasoning_chain(question, question_graph_result)
    print(answer)


def test_llm_using_chains():
    llm = LLMInteraction()
    answer = llm.get_final_answer(final_reasoning)
    print(answer)
    assert answer.answer == question.correct_answer
