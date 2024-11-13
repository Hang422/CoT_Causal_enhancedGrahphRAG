from src.modules.MedicalQuestion import MedicalQuestion
from src.graphrag.entity_processor import EntityProcessor
from tests.example import questions
from pathlib import Path

question = MedicalQuestion.from_cache(Path('../cache/original'), str(questions[0].question), questions[0].options)
processor = EntityProcessor()


def test_text():
    assert len(processor.process_text(question.question)) > 0
    assert len(processor.process_text(question.options.get('opa'))) > 0


def test_pairs():
    assert len(processor.process_entity_pairs(question)) > 0


def test_batch_convert():
    cuis = ['C0032961', 'C0043210', 'C0027750', 'C0680063', 'C0150600', 'C0013080']
    assert len(processor.batch_get_names(cuis)) == len(cuis)
    assert set(cuis) == set(processor.batch_get_cuis(processor.batch_get_names(cuis)))
