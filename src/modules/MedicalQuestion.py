from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import logging
import json
from pathlib import Path
import hashlib
import pandas as pd


@dataclass
class MedicalQuestion:
    """Medical question with all necessary information"""
    # Basic information
    question: str
    topic_name: str
    options: Dict[str, str]  # 使用 'opa', 'opb', 'opc', 'opd' 作为键
    correct_answer: Optional[str] = None  # 0-3 对应 A-D

    # Casual knowledge elements
    casual_nodes: Dict[str, List[List[str]]] = None  # [a:[[a,b][c,d]],b:[[a,b][c,d]]]
    casual_relationships: Dict[str, List[List[str]]] = None  # [a:[cause],b:[affect]]
    casual_paths: Dict[str, List[str]] = None  # [a:"A-cause->B", b:"C-affect->D"]
    casual_paths_nodes_refine: Dict[str, List] = None

    casual_relationships_refine: Dict[str, List[List[str]]] = None  # [a:[cause],b:[affect]]
    CG_nodes: List[str] = None
    CG_relationships: List[List[str]] = None
    CG_paths: List[str] = None

    # Knowledge elements
    entities_original_pairs: Dict[str, List] = None  # 'start'=[a,b,c] 'end' = [c,d,e]
    KG_nodes: List[List[str]] = None  # [[a,x,c][b,x,d][c,x,e]]
    KG_relationships: List[List[str]] = None  # [[cause,cause],[affect,affect][affect,affect]]
    KG_paths: List[str] = None  # ["A-cause->B-affect->C"]

    # Reasoning and answer
    reasoning: Optional[str] = None
    answer: Optional[str] = None  # 0-3 对应 A-D
    confidence: Optional[float] = None

    cache_dir = None

    def __post_init__(self):
        # 设置缓存目录
        self.cache_dir = Path("../../cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化列表类型的属性
        if self.casual_nodes is None:
            self.casual_nodes = {'opa': [], 'opb': [], 'opc': [], 'opd': []}
        if self.casual_relationships is None:
            self.casual_relationships = {'opa': [], 'opb': [], 'opc': [], 'opd': []}
        if self.casual_paths is None:
            self.casual_paths = {'opa': [], 'opb': [], 'opc': [], 'opd': []}
        if self.entities_original_pairs is None:
            self.entities_original_pairs = {'start': [], 'end': []}
        if self.KG_nodes is None:
            self.KG_nodes = []
        if self.KG_relationships is None:
            self.KG_relationships = []
        if self.KG_paths is None:
            self.KG_paths = []
        if self.casual_paths_nodes_refine is None:
            self.casual_paths_nodes_refine = {'start': [], 'end': []}
        if self.casual_relationships_refine is None:
            self.casual_relationships_refine = {'start': [], 'end': []}


    def generate_paths(self) -> None:
        """生成人类可读的路径表示"""
        # 为每个选项生成因果路径
        for option in ['opa', 'opb', 'opc', 'opd']:
            self.casual_paths[option] = []
            if option in self.casual_nodes and option in self.casual_relationships:
                for nodes, rels in zip(self.casual_nodes[option], self.casual_relationships[option]):
                    path = ""
                    for i, node in enumerate(nodes):
                        path += f"({node})"
                        if i < len(rels):
                            path += f"-{rels[i]}->"
                    self.casual_paths[option].append(path)

        # 生成知识图谱路径
        self.KG_paths = []
        for nodes, rels in zip(self.KG_nodes, self.KG_relationships):
            path = ""
            for i, node in enumerate(nodes):
                path += f"({node})"
                if i < len(rels):
                    path += f"-{rels[i]}->"
            self.KG_paths.append(path)
    @property
    def is_correct(self) -> bool:
        """检查答案是否正确"""
        return self.answer == self.correct_answer if self.answer is not None and self.correct_answer is not None else False

    def get_all_paths(self) -> Dict[str, List[str]]:
        """获取所有路径的字符串表示"""
        return {
            'casual_paths': self.casual_paths,
            'KG_paths': self.KG_paths
        }

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)

    def to_cache(self) -> None:
        """将问题数据缓存到本地"""
        # 使用问题文本和选项生成唯一标识
        cache_id = self._generate_cache_id()
        cache_path = self.cache_dir / f"{cache_id}.json"

        # 保存为JSON
        with cache_path.open('w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_cache(cls, cachepath, question_text: str, options: Dict[str, str]) -> Optional['MedicalQuestion']:
        """从缓存加载问题数据"""
        # 生成缓存ID
        cache_id = cls._generate_cache_id_static(question_text, options)
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

    def _generate_cache_id(self) -> str:
        """生成缓存文件的唯一标识"""
        content = self.question + ''.join(self.options.values())
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def _generate_cache_id_static(question_text: str, options: Dict[str, str]) -> str:
        """静态方法生成缓存ID"""
        content = question_text + ''.join(options.values())
        return hashlib.md5(content.encode()).hexdigest()

    def validate_paths(self) -> bool:
        """验证路径数据的一致性"""
        # 检查因果路径

        for nodes, rels in zip(self.casual_nodes, self.casual_relationships):
            if len(nodes) != len(rels) + 1:
                return False

        # 检查知识图谱路径
        if len(self.KG_nodes) != len(self.KG_relationships):
            return False
        for nodes, rels in zip(self.KG_nodes, self.KG_relationships):
            if len(nodes) != len(rels) + 1:
                return False

        # 检查生成的路径
        if len(self.casual_paths) != len(self.casual_nodes):
            return False
        if len(self.KG_paths) != len(self.KG_nodes):
            return False

        return True

    def has_complete_paths(self) -> bool:
        """检查是否有完整的路径数据"""
        return (
                bool(self.casual_paths) or
                bool(self.KG_paths)
        )

    def set_cache_paths(self, path) -> None:
        self.cache_dir = path
        self.cache_dir.mkdir(parents=True, exist_ok=True)

