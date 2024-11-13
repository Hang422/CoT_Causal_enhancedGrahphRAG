from typing import List

import src.graphrag.query_processor
from src.graphrag.query_processor import QueryProcessor
import pandas as pd
from tests.example import questions
from config import config
from pathlib import Path
from src.modules.MedicalQuestion import MedicalQuestion

question = MedicalQuestion.from_cache(Path('../cache/original'), str(questions[0].question), questions[0].options)


def test_query_casual_graph():
    config.set_database('casual')
    processor = QueryProcessor()
    processor.process_casual_paths(question)
    print(question.casual_paths)


def test_query_kg_graph():
    config.set_database('knowledge')
    processor = QueryProcessor()
    pair = {'start': ["Myelinated nerve fiber", "oncotic pressure of fluid leaving capillaries"], 'end': ["Impulse conduction speed", "glucose concentration in glomerular filtrate"]}
    question.entities_original_pairs = pair
    processor.process_entity_pairs(question)
    print(question.KG_paths)
