from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import logging
import json
from pathlib import Path
import hashlib

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SubGraph:

    # Path components
    nodes: List[List[str]] = None  # [[a,x,c], [b,x,d], [c,x,e]]
    relationships: List[List[str]] = None  # [[cause,cause], [affect,affect], [affect,affect]]
    paths: List[str] = None  # ["A-cause->B-affect->C"]

    def clear(self):
        self.nodes.clear()
        self.relationships.clear()
        self.paths.clear()

    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""
        return {
            'nodes': self.nodes,
            'relationships': self.relationships,
            'paths': self.paths if isinstance(self.paths, list) else []
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SubGraph':
        """从字典创建实例"""
        if data is None:
            return cls()
        return cls(
            nodes=data.get('nodes', []),
            relationships=data.get('relationships', []),
            paths=data.get('paths', [])
        )

    def __post_init__(self):
        """Initialize empty collections if None"""
        if self.nodes is None:
            self.nodes = []
        if self.relationships is None:
            self.relationships = []
        if self.paths is None:
            self.paths = []

    def generate_paths(self) -> None:
        """Generate human-readable path representations"""
        self.paths = []
        for nodes, rels in zip(self.nodes, self.relationships):
            path = ""
            for i, node in enumerate(nodes):
                path += f"({node})"
                if i < len(rels):
                    path += f"-{rels[i]}->"
            self.paths.append(path)

    def validate(self) -> bool:
        """Validate path data consistency"""
        # Check if we have nodes and relationships
        if len(self.nodes) != len(self.relationships):
            return False

        # Check if each path has correct number of nodes and relationships
        for nodes, rels in zip(self.nodes, self.relationships):
            if len(nodes) != len(rels) + 1:
                return False

        return True


@dataclass
class MedicalQuestion:
    """Medical question with all necessary information"""
    # Basic information
    question: str
    is_multi_choice: bool
    correct_answer: str  # 0-3 对应 A-D, or yes and no

    options: Optional[Dict[str, str]] = None  # 使用 'opa', 'opb', 'opc', 'opd' 作为键
    topic_name: Optional[str] = None
    context: Optional[str] = None

    # Casual knowledge elements
    initial_causal_graph: SubGraph = None
    causal_graph: SubGraph = None
    knowledge_graph: SubGraph = None
    enhanced_graph: SubGraph = None

    reasoning_chain: Optional[List[str]] = None
    enhanced_information: Optional[str] = None  # 改为使用 Optional[str]

    chain_coverage : Optional[Dict] = None
    # Reasoning and answer
    analysis: Optional[str] = None
    answer: Optional[str] = None  # 0-3 对应 A-D or yes and no
    confidence: Optional[float] = None

    # Search result of normal graphRAG
    normal_results: Optional[List[str]] = None

    cache_dir = None

    def __post_init__(self):
        """初始化后处理"""
        # 设置缓存目录
        self.cache_dir = Path("../../cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.causal_graph is None:
            self.causal_graph = SubGraph()
        if self.knowledge_graph is None:
            self.knowledge_graph = SubGraph()
        if self.initial_causal_graph is None:
            self.initial_causal_graph = SubGraph()
        if self.enhanced_graph is None:
            self.enhanced_graph = SubGraph()

        # 通用属性初始化
        if self.enhanced_information is None:
            self.enhanced_information = ""
        if self.reasoning_chain is None:
            self.reasoning_chain = []
        if self.chain_coverage is None:
            self.chain_coverage = {
                'success_counts': [],
                'coverage_rates': [],
                'total_successes': 0
            }
        if self.normal_results is None:
            self.normal_results = []

    def generate_paths(self) -> None:
        """生成人类可读的路径表示"""
        # 为每个选项生成因果路径
        self.initial_causal_graph.paths.clear()
        self.causal_graph.paths.clear()
        self.knowledge_graph.paths.clear()
        self.enhanced_graph.paths.clear()
        self.initial_causal_graph.generate_paths()
        self.causal_graph.generate_paths()
        self.knowledge_graph.generate_paths()
        self.enhanced_graph.generate_paths()

    @property
    def is_correct(self) -> bool:
        """检查答案是否正确"""
        if not self.answer:
            return False
        return self.answer.lower() == self.correct_answer.lower()

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        data = {
            'question': self.question,
            'is_multi_choice': self.is_multi_choice,
            'correct_answer': self.correct_answer,
            'options': self.options,
            'topic_name': self.topic_name,
            'context': self.context,
            'initial_causal_graph': self.initial_causal_graph.to_dict() if self.initial_causal_graph else None,
            'causal_graph': self.causal_graph.to_dict() if self.causal_graph else None,
            'knowledge_graph': self.knowledge_graph.to_dict() if self.knowledge_graph else None,
            'enhanced_graph': self.enhanced_graph.to_dict() if self.enhanced_graph else None,
            'reasoning_chain': self.reasoning_chain,
            'enhanced_information': self.enhanced_information,
            'analysis': self.analysis,
            'answer': self.answer,
            'confidence': self.confidence,
            'chain_coverage': self.chain_coverage,
            'normal_results': self.normal_results
        }
        return data

    @classmethod
    def from_cache(cls, cachepath, question_text: str) -> Optional['MedicalQuestion']:
        """从缓存加载问题数据"""
        cache_id = hashlib.md5(question_text.encode()).hexdigest()
        cache_path = cachepath / f"{cache_id}.json"

        if cache_path.exists():
            try:
                with cache_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)

                # 处理 SubGraph 对象
                if 'initial_causal_graph' in data:
                    data['initial_causal_graph'] = SubGraph.from_dict(data['initial_causal_graph'])
                if 'causal_graph' in data:
                    data['causal_graph'] = SubGraph.from_dict(data['causal_graph'])
                if 'knowledge_graph' in data:
                    data['knowledge_graph'] = SubGraph.from_dict(data['knowledge_graph'])
                if 'enhanced_graph' in data:
                    data['enhanced_graph'] = SubGraph.from_dict(data['enhanced_graph'])
                if 'reasoning_chain' in data:
                    if isinstance(data['reasoning_chain'], list):
                        data['reasoning_chain'] = data['reasoning_chain']
                    else:
                        data['reasoning_chain'] = list(data['reasoning_chain'])

                # 处理 chain_coverage，如果缓存中没有这个字段，设置默认值
                if 'chain_coverage' not in data:
                    data['chain_coverage'] = {
                        'success_counts': [],
                        'coverage_rates': [],
                        'total_successes': 0
                    }

                return cls(**data)
            except Exception as e:
                logging.error(f"Error loading cache: {str(e)}")
                return None
        return None

    def to_cache(self) -> None:
        """将问题数据缓存到本地"""
        # 使用问题文本和选项生成唯一标识
        cache_id = hashlib.md5(self.question.encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_id}.json"

        # 保存为JSON
        with cache_path.open('w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def set_cache_paths(self, path) -> None:
        self.cache_dir = path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
