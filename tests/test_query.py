from typing import List

from src.graphrag.query_processor import QueryProcessor
import pandas as pd
from src.llm.interactor import LLMInteraction
from src.modules.data_format import Question, OptionKey
from tests.example import question, question_graph_result, question_analysis
from src.graphrag.entity_processor import EntityProcessor
from src.llm.interactor import LLMInteraction
from config import config

entity_processor = EntityProcessor()
questionCUIs = entity_processor.process_question(question)


def test_query_casual_graph():
    processor = QueryProcessor()
    casual_graph = processor.query_question_cuis(questionCUIs)
    print(casual_graph)


def test_query_entities():
    config.set_database("knowledge")
    processor = QueryProcessor()

    # One-liner version
    pairs_with_paths = [(pair, processor.query_entity_pairs(pair))
                        for pair in question_analysis.entity_pairs]

    print(pairs_with_paths)



