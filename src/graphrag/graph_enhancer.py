from typing import List, Tuple, Dict
from collections import defaultdict
from neo4j import GraphDatabase
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from config import config
from src.modules.MedicalQuestion import MedicalQuestion
from src.graphrag.entity_processor import EntityProcessor


class PathMerger:
    def __init__(self):
        self.debug = False


def merge_group(paths: List[str]) -> str:
    """合并一组路径"""
    parts_by_position = []
    example_path = paths[0]
    path_parts = example_path.split('->')

    # 初始化每个位置的实体集合
    for _ in range(len(path_parts)):
        parts_by_position.append(set())

    # 收集每个位置的所有实体
    for path in paths:
        current_parts = path.split('->')
        for i, part in enumerate(current_parts):
            if i == 0:  # 起始实体
                parts_by_position[i].add(part)
            else:
                # 处理中间实体
                entity = part.split('-')[0] if '-' in part else part
                parts_by_position[i].add(entity)

    # 构建合并后的路径
    merged_parts = []
    for i, parts in enumerate(parts_by_position):
        if i == 0:  # 起始实体
            merged_parts.append(next(iter(parts)))
        elif i == len(parts_by_position) - 1:  # 终止实体
            merged_parts.append(next(iter(parts)))
        else:
            # 处理中间实体，合并括号内的内容
            entities = set()
            for part in parts:
                entity = part.strip('()')
                entities.add(entity)
            relation = example_path.split('->')[i].split('-')[1]
            merged_entity = f"({' and '.join(sorted(entities))})"
            merged_parts.append(f"{merged_entity}-{relation}")

    return '->'.join(merged_parts)


class GraphEnhancer:

    def __init__(self):
        """Initialize processor with database configuration"""
        self.logger = logging.getLogger(__name__)
        self.entity_processor = EntityProcessor()  # Still needed for name-to-CUI conversion

    def extract_path_elements(self, path: str):
        """提取路径的起点、终点和关系"""
        parts = path.split('->')
        start_entity = parts[0].strip('()')
        end_entity = parts[-1].strip('()')
        # 提取关系序列
        relations = []
        for i in range(1, len(parts)):
            if '-' in parts[i]:
                rel = parts[i].split('-')[1]
                relations.append(rel)

        return start_entity, end_entity, tuple(relations)

    def group_mergeable_paths(self, paths: List[str]):
        """将可合并的路径分组"""
        groups = {}
        for path in paths:
            start, end, relations = self.extract_path_elements(path)
            key = (start, end, relations)
            if key not in groups:
                groups[key] = []
            groups[key].append(path)
        return groups

    def merge_paths(self, paths: List[str]) -> List[str]:
        """合并路径"""
        if not paths:
            return []

        # 分组可合并的路径
        groups = self.group_mergeable_paths(paths)

        # 处理每个组
        merged_paths = []
        for key, group_paths in groups.items():
            if len(group_paths) > 1:
                merged = merge_group(group_paths)
                merged_paths.append(merged)
            else:
                merged_paths.extend(group_paths)

        return merged_paths

    def clear_paths(self, question: MedicalQuestion):
        # 构建推理链的CUI序列
        all_chains = []
        for chain in question.reasoning_chain:
            cuis_chain = []
            steps = chain.split('->')
            for step in steps:
                cuis = self.entity_processor.process_text(step)
                cuis_chain.append(cuis)
            all_chains.append(cuis_chain)

        # 收集所有路径的节点和关系
        nodes_list = []
        relationships_list = []
        if len(question.initial_causal_graph.nodes) > 0:
            nodes_list.append(question.initial_causal_graph.nodes)
            relationships_list.append(question.initial_causal_graph.relationships)

        valid_paths = []
        valid_nodes = []
        valid_relationships = []

        # 遍历每条路径（由一组节点和关系组成）
        for nodes_groups, relationships_groups in zip(nodes_list, relationships_list):

            # 获取这条路径的起点和终点的CUI
            for path_nodes, path_relationships in zip(nodes_groups, relationships_groups):

                start_cui = self.entity_processor.get_name_cui(path_nodes[0])
                end_cui = self.entity_processor.get_name_cui(path_nodes[-1])

                # 检查每条推理链
                for chain_idx, chain in enumerate(all_chains):
                    start_found = False
                    end_found = False
                    start_pos = -1
                    end_pos = -1

                    # 在推理链中查找起点和终点
                    flag = True
                    for pos, step_cuis in enumerate(chain):
                        if start_cui in step_cuis and flag:
                            start_found = True
                            start_pos = pos
                            flag = False
                        if end_cui in step_cuis:
                            end_found = True
                            end_pos = pos

                    # 验证路径是否有效
                    if start_found and end_found and start_pos < end_pos:
                        # 构建路径字符串
                        path = ""
                        for i, node in enumerate(path_nodes):
                            path += f"({node})"
                            if i < len(path_relationships):
                                path += f"-{path_relationships[i]}->"

                        path_info = {
                            'path': path,
                            'chain_index': chain_idx,
                            'start_position': start_pos,
                            'end_position': end_pos,
                            'start_cui': start_cui,
                            'end_cui': end_cui
                        }
                        valid_paths.append(path_info)
                        valid_nodes.append(path_nodes)
                        valid_relationships.append(path_relationships)
                        break

        # 按链索引和位置排序
        indices = list(range(len(valid_paths)))
        indices.sort(key=lambda i: (valid_paths[i]['chain_index'],
                                    valid_paths[i]['start_position'],
                                    valid_paths[i]['end_position']))

        # 重排所有列表
        # 重排所有列表

        valid_paths = [valid_paths[i] for i in indices]
        valid_nodes = [valid_nodes[i] for i in indices]
        valid_relationships = [valid_relationships[i] for i in indices]

        # 更新enhanced_graph
        question.enhanced_graph.nodes = valid_nodes
        question.enhanced_graph.relationships = valid_relationships
        question.enhanced_graph.generate_paths()  # 自动生成paths

        question.enhanced_graph.paths.extend(question.causal_graph.paths)
        question.enhanced_graph.paths.extend(question.knowledge_graph.paths)

        question.enhanced_graph.paths = self.merge_paths(question.enhanced_graph.paths)

        return valid_paths


# 测试
if __name__ == "__main__":
    test_paths = [
        "(Phosphorus)-TREATS->(Bacteria)-CAUSES->(Dental Plaque)",
        "(Phosphorus)-TREATS->(Inflammation)-CAUSES->(Dental Plaque)",
        "(Phosphorus)-TREATS->(Microbial Biofilms)-CAUSES->(Dental Plaque)",
        "(potassium)-TREATS->(Microbial Biofilms)-CAUSES->(Dental Plaque)",
        "(potassium)-TREATS->(Bacteria)-CAUSES->(Dental Plaque)",
        "(potassium)-TREATS->(Inflammation)-CAUSES->(Dental Plaque)",
        "(calcium)-CAUSES->(Inflammation)-CAUSES->(Dental Plaque)",
        "(calcium)-TREATS->(Periodontitis)-CAUSES->(Dental Plaque)",
        "(calcium)-TREATS->(Microbial Biofilms)-CAUSES->(Dental Plaque)"
    ]

    enhancer = GraphEnhancer()

    # 检查每个实体的CUI转换
    print("\n=== 测试CUI转换 ===")
    test_entities = [
        "Phosphorus", "Bacteria", "Dental Plaque",
        "Metronidazole", "Infection", "Painful micturition",
        "Condition"
    ]

    """print("实体到CUI的映射:")
    for entity in test_entities:
        cui = enhancer.entity_processor.get_name_cui(entity)
        print(f"{entity}: {cui}")"""

    # 创建测试用的MedicalQuestion对象
    path = config.paths["cache"] / 'test1' / 'enhanced'
    test_question = MedicalQuestion.from_cache(path,
                                               "In heavy calculus formers, early plaque consists of Ca, PO4 and k in what proportions:")
    test_question.reasoning_chain = ["Calcium + Inflammation -> Dental Plaque -> Calcium > phosphorous > potassium",
    "Phosphorous + Bacteria -> Dental Plaque -> Phosphorous > calcium > potassium",
    "Potassium + Microbial Biofilms -> Dental Plaque -> Potassium > phosphorous > calcium"]

    print("\n初始路径:")
    print("Initial Causal Graph paths:", test_question.initial_causal_graph.paths)
    print("Causal Graph paths:", test_question.causal_graph.paths)

    print("\n推理链和对应的CUIs:")
    for i, chain in enumerate(test_question.reasoning_chain):
        print(f"\nChain {i}: {chain}")
        steps = chain.split(" -> ")
        print("步骤的CUIs:")
        for step in steps:
            cuis = enhancer.entity_processor.process_text(step)
            print(f"  {step}: {cuis}")

    # 执行路径验证
    valid_paths = enhancer.clear_paths(test_question)

    print("\n验证后的路径:")
    if valid_paths:
        for path_info in valid_paths:
            print(f"\n路径: {path_info['path']}")
            print(f"匹配的推理链索引: {path_info['chain_index']}")
            print(f"起始位置: {path_info['start_position']} -> {path_info['end_position']}")
            print(f"CUIs: {path_info['start_cui']} -> {path_info['end_cui']}")
    else:
        print("没有找到有效的路径")

    print("\n更新后的问题对象中的路径:")
    print("Causal Graph paths:", test_question.enhanced_graph.paths)
