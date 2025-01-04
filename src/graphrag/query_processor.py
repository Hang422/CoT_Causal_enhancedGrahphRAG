from neo4j import GraphDatabase
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from config import config
from src.modules.MedicalQuestion import MedicalQuestion
from src.graphrag.entity_processor import EntityProcessor
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import CohereEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever


class QueryProcessor:
    """Neo4j query processor for medical question path finding"""

    def __init__(self):
        """Initialize processor with database configuration"""
        self.logger = logging.getLogger(__name__)
        db_config = config.get_db_config()

        self.driver = GraphDatabase.driver(
            db_config["uri"],
            auth=(db_config["username"], db_config["password"]),
            max_connection_lifetime=3600,
            max_connection_pool_size=db_config["max_connections"],
        )
        self.entity_processor = EntityProcessor()  # Still needed for name-to-CUI conversion

        self.embedder = CohereEmbeddings(
            model=config.cohere["model"],
            api_key=config.cohere["api_key"]
        )
        self.vector_retriever = VectorRetriever(
            driver=self.driver,
            index_name="medical_vector_index",
            embedder=self.embedder,
            neo4j_database='neo4j'
        )

        self.hyper_parameter_cg = 3
        self.hyper_parameter_kg = 2

    def close(self):
        """Close database connection"""
        if hasattr(self, 'driver'):
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def process_causal_graph_paths_enhanced(self, start_cui: str, end_cui: str) -> List[Dict]:
        """Query and process paths in the causal graph."""
        query = """
        MATCH path = allShortestPaths((start)-[*..5]->(end))
        WHERE start.CUI = $start_cui AND end.CUI = $end_cui
        RETURN 
            [node IN nodes(path) | node.Name] AS node_names,
            [rel IN relationships(path) | type(rel)] AS relationships,
            [rel IN relationships(path) | COALESCE(toFloat(rel.Strength), 0.0)] AS scores,
            length(path) AS path_length
        ORDER BY path_length ASC
        LIMIT 10
        """

        with self.driver.session(database='Causal') as session:
            result = session.run(query, start_cui=start_cui, end_cui=end_cui)
            paths = []

            for record in result:
                paths.append({
                    'node_names': record['node_names'],
                    'relationships': record['relationships'],
                    'scores': record['scores'],
                    'path_length': record['path_length']
                })

            # 删除路径长度大于最短路径长度+1的路径
            if paths:
                shortest_length = paths[0]['path_length']
                paths = [path for path in paths if path['path_length'] <= shortest_length + 1]

            # 对每组相同节点的路径，根据分数选择最高分路径
            unique_paths = {}
            for path in paths:
                key = tuple(path['node_names'])
                score = sum(path['scores']) / (1 + path['path_length'])  # 打分逻辑

                if key not in unique_paths or unique_paths[key]['score'] < score:
                    unique_paths[key] = {
                        'node_names': path['node_names'],
                        'relationships': path['relationships'],
                        'path_length': path['path_length'],
                        'score': score
                    }

            # 返回处理后的路径
            sorted_paths = sorted(unique_paths.values(), key=lambda x: x['score'], reverse=True)

            return sorted_paths[:self.hyper_parameter_cg]

    def process_knowledge_graph_paths(self, start_cui: str, end_cui: str) -> List[Dict]:
        """Query and process paths in the knowledge graph."""
        query = """
        MATCH path = allShortestPaths((start)-[*..5]->(end))
        WHERE start.CUI = $start_cui AND end.CUI = $end_cui
        RETURN 
            [node IN nodes(path) | node.Name] AS node_names,
            [rel IN relationships(path) | type(rel)] AS relationships,
            length(path) AS path_length
        ORDER BY path_length ASC
        LIMIT 10
        """

        with self.driver.session(database='Knowledge') as session:
            result = session.run(query, start_cui=start_cui, end_cui=end_cui)
            paths = []

            for record in result:
                paths.append({
                    'node_names': record['node_names'],
                    'relationships': record['relationships'],
                    'path_length': record['path_length']
                })

            # 直接返回最短的3条路径
            return paths[:self.hyper_parameter_kg]

    def generate_initial_causal_graph(self, question: MedicalQuestion) -> None:
        """Enhanced version of process_casual_paths that returns multiple shorter paths"""
        try:
            with self.driver.session(database='causal') as session:
                # 获取问题中的CUIs
                question_cuis = set(self.entity_processor.extract_cuis_from_text(question.question))
                keys = list(question.options.keys())

                # 分别获取每个选项的CUIs
                options_cuis = set()
                for key, option_text in question.options.items():  # 直接遍历选项字典
                    processed_cuis = self.entity_processor.extract_cuis_from_text(option_text)
                    options_cuis.update(set(processed_cuis))  # 使用 update 更新集合

                # 用于去重的集合
                seen_paths = set()

                for start_cui in options_cuis:
                    for end_cui in question_cuis:
                        if start_cui == end_cui:
                            continue

                        paths = self.process_causal_graph_paths_enhanced(start_cui, end_cui)
                        for path in paths:
                            # 创建用于去重的路径标识
                            path_identifier = (
                                tuple(path['node_names']),
                                tuple(path['relationships'])
                            )

                            # 如果是新路径，添加到结果中
                            if path_identifier not in seen_paths:
                                seen_paths.add(path_identifier)
                                question.initial_causal_graph.nodes.append(path['node_names'])
                                question.initial_causal_graph.relationships.append(path['relationships'])

                for start_cui in question_cuis:
                    for end_cui in options_cuis:
                        if start_cui == end_cui:
                            continue

                        paths = self.process_causal_graph_paths_enhanced(start_cui, end_cui)
                        for path in paths:
                            # 创建用于去重的路径标识
                            path_identifier = (
                                tuple(path['node_names']),
                                tuple(path['relationships'])
                            )

                            # 如果是新路径，添加到结果中
                            if path_identifier not in seen_paths:
                                seen_paths.add(path_identifier)
                                question.initial_causal_graph.nodes.append(path['node_names'])
                                question.initial_causal_graph.relationships.append(path['relationships'])

        except Exception as e:
            self.logger.error(f"Error in enhanced casual paths processing: {str(e)}", exc_info=True)

        question.generate_paths()  # Update path strings

    def process_chain_of_thoughts(self, question: MedicalQuestion, graph: str, enhancement: bool) -> None:
        """处理思维链的路径检索，并计算相对覆盖率"""
        question.causal_graph.clear()
        question.knowledge_graph.clear()
        chain_success_counts = []  # 存储每条链的查询成功次数

        for chain in question.reasoning_chain:
            success_count = 0  # 每条链的查询成功次数

            # 分割思维链的步骤
            steps = chain.split('->')
            # 遍历相邻步骤对
            for i in range(len(steps) - 1):
                start = steps[i].strip()
                end = steps[i + 1].strip()
                # print(f"start {start} end {end}")
                # 去掉置信度部分（如果存在）
                if '%' in end:
                    end = end.split('%')[0].strip()

                # 通过entity processor获取CUI
                start_cuis = self.entity_processor.extract_cuis_from_text(start)
                end_cuis = self.entity_processor.extract_cuis_from_text(end)

                if len(start_cuis) == 0 or len(end_cuis) == 0:
                    continue

                for start_cui in start_cuis:
                    is_success = False
                    if is_success:
                        break
                    for end_cui in end_cuis:
                        if is_success:
                            continue

                        if start_cui == end_cui:
                            continue

                        if start_cui and end_cui:
                            # print(f"start {start_cui} end {end_cui}")
                            if graph == 'both':
                                paths = self.process_causal_graph_paths_enhanced(start_cui, end_cui)
                                if len(paths) > 0:
                                    success_count += 1
                                    is_success = True
                                    for path in paths:
                                        question.causal_graph.nodes.append(path['node_names'])
                                        question.causal_graph.relationships.append(path['relationships'])
                                else:
                                    paths = self.process_knowledge_graph_paths(start_cui, end_cui)
                                    if len(paths) > 0:
                                        success_count += 1
                                        is_success = True
                                        for path in paths:
                                            question.knowledge_graph.nodes.append(path['node_names'])
                                            question.knowledge_graph.relationships.append(path['relationships'])

                            elif graph == 'knowledge':
                                paths = self.process_knowledge_graph_paths(start_cui, end_cui)
                                if len(paths) > 0:
                                    success_count += 1
                                    is_success = True
                                    for path in paths:
                                        question.knowledge_graph.nodes.append(path['node_names'])
                                        question.knowledge_graph.relationships.append(path['relationships'])

                            elif graph == 'causal':
                                paths = self.process_causal_graph_paths_enhanced(start_cui, end_cui)
                                if len(paths) > 0:
                                    success_count += 1
                                    is_success = True
                                    for path in paths:
                                        question.causal_graph.nodes.append(path['node_names'])
                                        question.causal_graph.relationships.append(path['relationships'])

            chain_success_counts.append(success_count)

        # 计算总查询成功次数
        total_successes = sum(chain_success_counts)

        # 计算每条链的相对覆盖率
        if total_successes > 0:
            chain_coverage_rates = [(count / total_successes) * 100 for count in chain_success_counts]
        else:
            chain_coverage_rates = [0.0] * len(chain_success_counts)

        # 存储覆盖率信息
        question.chain_coverage = {
            'success_counts': chain_success_counts,
            'coverage_rates': chain_coverage_rates,
            'total_successes': total_successes
        }

        question.generate_paths()

    def process_vector_search(self, question: MedicalQuestion) -> None:
        """
        对问题和选项进行向量检索，返回最相关的结果

        Args:
            question: MedicalQuestion对象，包含问题和选项信息
        """
        try:
            # 存储所有检索结果
            all_results = set()

            # 为每个选项构建查询文本并执行检索
            for option_key, option_text in question.options.items():
                query_text = f"{question.question} {option_text}"
                results = self.vector_retriever.search(query_text=query_text, top_k=3)

                # 将结果添加到集合中（去重）
                for result in results.items:
                    result_dict = eval(result.content)
                    text = result_dict.get('text', '')
                    if text:
                        all_results.add(text)

            # 存储结果到问题对象中
            question.normal_results = list(all_results)

        except Exception as e:
            self.logger.error(f"Error in vector search: {str(e)}")
            question.normal_results = []


def test_path_query_consistency():
    processor = QueryProcessor()
    entity_processor = EntityProcessor()
    print(list(entity_processor.extract_cuis_from_text('CHAIN: "Substantia nigra"'))[0],list(entity_processor.extract_cuis_from_text('dopamine production'))[0])
    print(processor.process_knowledge_graph_paths(list(entity_processor.extract_cuis_from_text('CHAIN: "Substantia nigra"'))[0],list(entity_processor.extract_cuis_from_text('dopamine production'))[0]))
    print(list(entity_processor.extract_cuis_from_text('Substantia nigra structure'))[0],
          list(entity_processor.extract_cuis_from_text('Dopamine'))[0])
    print(processor.process_knowledge_graph_paths(
        list(entity_processor.extract_cuis_from_text('Substantia nigra structure'))[0],
        list(entity_processor.extract_cuis_from_text('Dopamine'))[0]))
