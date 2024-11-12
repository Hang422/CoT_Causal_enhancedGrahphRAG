from src.graphrag.entity_processor import EntityProcessor
from src.modules.data_format import Question, OptionKey, EntityPairs
import pandas as pd
from tests.example import question, question_analysis

def test_entity_processor():
    entity_processor = EntityProcessor()
    assert entity_processor.process_question(question) is not None


def test_pairs_processor():
    entity_processor = EntityProcessor()
    for pair in question_analysis.entity_pairs:
        print(entity_processor.process_entity_pairs(pair))

