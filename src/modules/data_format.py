# data_format.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class OptionKey(str, Enum):
    """Question option identifiers"""
    A = "a"
    B = "b"
    C = "c"
    D = "d"


class DataFormat:
    @dataclass
    class Question:
        """Initial question format"""
        text: str
        options: Dict[OptionKey, str]
        correct_answer: Optional[OptionKey] = None

        def __post_init__(self):
            self.options = {
                OptionKey(k) if isinstance(k, str) else k: v
                for k, v in self.options.items()
            }

    @dataclass
    class EntityInfo:
        """Entity with UMLS information"""
        name: str
        types: List[str]
        cui: str

    @dataclass
    class QuestionCUIs:
        """Question and options converted to CUIs"""
        question_cuis: List[str]
        question_entities: List['DataFormat.EntityInfo']
        option_cuis: Dict[OptionKey, List[str]]
        option_entities: Dict[OptionKey, List['DataFormat.EntityInfo']]

    @dataclass
    class GraphPath:
        """A path in the knowledge graph"""
        nodes: List[str]  # CUIs
        node_names: List[str]  # Entity names
        relationships: List[str]
        source_entity: Optional['DataFormat.EntityInfo'] = None
        target_entity: Optional['DataFormat.EntityInfo'] = None

    @dataclass
    class QuestionGraphResult:
        """Results of causal path search"""
        question_paths: List['DataFormat.GraphPath']  # 问题实体内部路径
        option_paths: Dict[OptionKey, List['DataFormat.GraphPath']]  # 问题到每个选项的路径

    @dataclass
    class EntityPairs:
        """Entity pair requiring verification"""
        start: List[str]
        end: List[str]
        reasoning: str

    @dataclass
    class QuestionAnalysis:
        """LLM's initial analysis"""
        reasoning_chain: List[str]  # 推理步骤
        entity_pairs: List['DataFormat.EntityPairs']  # 需要验证的实体对

    @dataclass
    class FinalReasoning:
        """Final analysis combining all information for LLM"""
        question: 'DataFormat.Question'
        initial_paths: List['DataFormat.GraphPath']  # 初始因果路径
        entity_pairs: List[tuple['DataFormat.EntityPairs', List['DataFormat.GraphPath']]]  # 实体对及其验证路径
        reasoning_chain: List[str]  # 当前的推理链

    @dataclass
    class Answer:
        """LLM's final answer"""
        question: 'DataFormat.Question'
        answer: OptionKey
        confidence: float
        explanation: str
        isCorrect: bool = False

    @dataclass
    class EntitiesPairsCUIs:
        start: List[str]
        end: List[str]
        reasoning: str


Question = DataFormat.Question
QuestionCUIs = DataFormat.QuestionCUIs
GraphPath = DataFormat.GraphPath
QuestionGraphResult = DataFormat.QuestionGraphResult
EntityPairs = DataFormat.EntityPairs
FinalReasoning = DataFormat.FinalReasoning
QuestionAnalysis = DataFormat.QuestionAnalysis
EntityInfo = DataFormat.EntityInfo
Answer = DataFormat.Answer
EntityPairsCUIs = DataFormat.EntitiesPairsCUIs
