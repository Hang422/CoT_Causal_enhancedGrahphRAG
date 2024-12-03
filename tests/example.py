from src.modules.MedicalQuestion import MedicalQuestion
from src.graphrag.entity_processor import EntityProcessor
import pandas as pd

questions = []
df = pd.read_csv('../data/Construction/testdata/samples.csv')
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