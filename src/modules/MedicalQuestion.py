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
    # Entity pairs for path finding
    entities_pairs: Dict[str, List[str]] = None  # 'start'=[a,b,c] 'end'=[c,d,e]

    # Path components
    nodes: List[List[str]] = None  # [[a,x,c], [b,x,d], [c,x,e]]
    relationships: List[List[str]] = None  # [[cause,cause], [affect,affect], [affect,affect]]
    paths: List[str] = None  # ["A-cause->B-affect->C"]

    def __post_init__(self):
        """Initialize empty collections if None"""
        if self.entities_pairs is None:
            self.entities_pairs = {'start': [], 'end': []}
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

        # Check entity pairs
        if not self.entities_pairs.get('start') or not self.entities_pairs.get('end'):
            return False

        if len(self.entities_pairs['start']) != len(self.entities_pairs['end']):
            return False

        return True


@dataclass
class MedicalQuestion:
    """Medical question with all necessary information"""
    # Basic information
    question: str
    is_multi_choice: bool
    correct_answer: str  # 0-3 对应 A-D, or yes and no

    options: Optional[Dict[str, str]] = None # 使用 'opa', 'opb', 'opc', 'opd' 作为键
    topic_name: Optional[str] = None
    context: Optional[str] = None

    # Casual knowledge elements
    initial_causal_graph: SubGraph = None
    causal_graph: SubGraph = None
    knowledge_graph: SubGraph = None

    reasoning_chain: Optional[str] = None
    enhanced_information: Optional[str] = None  # 改为使用 Optional[str]

    # Reasoning and answer
    analysis: Optional[str] = None
    answer: Optional[str] = None  # 0-3 对应 A-D or yes and no
    confidence: Optional[float] = None

    cache_dir = None

    def __post_init__(self):
        """初始化后处理"""
        # 设置缓存目录
        self.cache_dir = Path("../../cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.causal_graph = SubGraph()
        self.knowledge_graph = SubGraph()
        self.initial_causal_graph = SubGraph()


        # 通用属性初始化
        if self.enhanced_information is None:
            self.enhanced_information = ""
        if self.reasoning_chain is None:
            self.reasoning_chain = ""

    def generate_paths(self) -> None:
        """生成人类可读的路径表示"""
        # 为每个选项生成因果路径

        self.initial_causal_graph.generate_paths()
        self.causal_graph.generate_paths()
        self.knowledge_graph.generate_paths()

        if len(self.initial_causal_graph.paths) == 0:
            self.initial_causal_graph.paths = "There is no obvious causal relationship."

        if len(self.causal_graph.paths) == 0:
            self.causal_graph.paths = "There is no obvious causal relationship."

    @property
    def is_correct(self) -> bool:
        """检查答案是否正确"""
        if not self.answer:
            return False
        return self.answer.lower() == self.correct_answer.lower()

    def get_all_paths(self) -> Dict[str, List[str]]:
        """获取所有路径的字符串表示"""
        return {
            'causal_paths': self.causal_graph.paths,
            'KG_paths': self.knowledge_graph.paths
        }

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)

    def to_cache(self) -> None:
        """将问题数据缓存到本地"""
        # 使用问题文本和选项生成唯一标识
        cache_id = hashlib.md5(self.question.encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_id}.json"

        # 保存为JSON
        with cache_path.open('w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_cache(cls, cachepath, question_text: str) -> Optional['MedicalQuestion']:
        """从缓存加载问题数据"""
        # 生成缓存ID
        cache_id = hashlib.md5(question_text.encode()).hexdigest()
        cache_path = cachepath / f"{cache_id}.json"

        if cache_path.exists():
            try:
                with cache_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                logging.error(f"Error loading cache: {str(e)}")
                return None
        return None

    def set_cache_paths(self, path) -> None:
        self.cache_dir = path
        self.cache_dir.mkdir(parents=True, exist_ok=True)