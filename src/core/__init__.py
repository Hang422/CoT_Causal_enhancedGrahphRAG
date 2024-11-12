from pipeline import ProcessingPipeline
from config import config
import pipeline

# 初始化处理流水线
pipeline = ProcessingPipeline(config, cache_enabled=True)

# 处理单个问题
question = pipeline.Question(
    text="What is the relationship between hypertension and stroke?",
    options={
        "a": "Hypertension increases stroke risk",
        "b": "No relationship",
        "c": "Inverse relationship",
        "d": "Unknown relationship"
    }
)
result = pipeline.process_question(question)

# 批量处理
questions = [question1, question2, question3]
results = pipeline.process_batch(
    questions,
    output_file=config.paths["output"] / "results.json"
)