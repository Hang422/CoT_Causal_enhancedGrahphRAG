from neo4j import GraphDatabase
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from config import config
from src.modules.MedicalQuestion import MedicalQuestion
from src.graphrag.entity_processor import EntityProcessor


def _find_shortest_path_score(session, start_cui: str, end_cui: str) -> List[Dict]:
    """Find highest scoring paths between two CUIs with length constraints"""
    query = """
    MATCH path = allShortestPaths((start)-[*..5]->(end))
    WHERE start.CUI = $start_cui AND end.CUI = $end_cui
    WITH path, length(path) as len
    ORDER BY len ASC
    WITH collect({
        nodes: [node in nodes(path) | node.Name],  -- Use node.Name instead of node.CUI
        rels: [rel in relationships(path) | rel.name],
        length: len
    }) as paths,
    min(len) as shortest_len
    UNWIND [p in paths WHERE p.length <= shortest_len + 1] as filtered_path
    RETURN 
        filtered_path.nodes as node_names,
        filtered_path.rels as relationships,
        filtered_path.length as path_length
    """

    relation_scores = {
        'CAUSES': 5, 'PRODUCES': 5, 'INDUCES': 5, 'induces': 5,
        'cause_of': 5, 'causative_agent_of': 5, 'has_causative_agent': 5,
        'STIMULATES': 4, 'INHIBITS': 4, 'AFFECTS': 4, 'DISRUPTS': 4,
        'positively_regulates': 4, 'negatively_regulates': 4, 'regulates': 4,
        'has_process_output': 3, 'has_result': 3, 'result_of': 3, 'CONVERTS_TO': 3,
        'PREVENTS': 2, 'TREATS': 2, 'AUGMENTS': 2,
        'INTERACTS_WITH': 1, 'ISA': 1, 'disease_has_finding': 1, 'MANIFESTATION_OF': 1
    }

    scored_paths = []
    result = session.run(query, start_cui=start_cui, end_cui=end_cui)

    for record in result:
        score = sum(relation_scores.get(rel, 1) for rel in record['relationships'])
        score *= 1.0 / (1 + record['path_length'])
        scored_paths.append({
            'node_names': record['node_names'],
            'relationships': record['relationships'],
            'path_length': record['path_length'],
            'score': score
        })

    scored_paths.sort(key=lambda x: x['score'], reverse=True)
    return scored_paths[:5]  # Return top 5 paths


def _find_shortest_path_enhance(session, start_cui: str, end_cui: str) -> List[Dict]:
    """Enhanced path finding that returns alternative shorter paths with scoring"""
    query = """
    // Get original shortest path and length
    MATCH (start {CUI: $start_CUI}), (end {CUI: $end_CUI})
    OPTIONAL MATCH initialPath = shortestPath((start)-[*..4]->(end))
    WITH start, end, 
         CASE WHEN initialPath IS NULL THEN -1 ELSE length(initialPath) END as original_length
    WHERE original_length > 0

    // Get neighbors and find shorter paths
    MATCH (start)-[r1]->(neighbor)
    WHERE neighbor <> end
    OPTIONAL MATCH alterPath = shortestPath((neighbor)-[*..3]->(end))
    WITH start, neighbor, r1, alterPath, original_length,
         CASE WHEN alterPath IS NULL THEN -1 ELSE length(alterPath) END as path_length
    WHERE alterPath IS NOT NULL AND path_length < original_length

    // Return complete path information including start node
    RETURN DISTINCT
           [start.cui] + [neighbor.cui] + [node in nodes(alterPath)[1..] | node.cui] as node_cuis,
           [r1.name] + [rel in relationships(alterPath) | rel.name] as relationships,
           path_length + 1 as total_length
    ORDER BY total_length ASC
    LIMIT 5
    """

    relation_scores = {
        'CAUSES': 5, 'PRODUCES': 5, 'INDUCES': 5, 'induces': 5,
        'cause_of': 5, 'causative_agent_of': 5, 'has_causative_agent': 5,
        'STIMULATES': 4, 'INHIBITS': 4, 'AFFECTS': 4, 'DISRUPTS': 4,
        'positively_regulates': 4, 'negatively_regulates': 4, 'regulates': 4,
        'has_process_output': 3, 'has_result': 3, 'result_of': 3, 'CONVERTS_TO': 3,
        'PREVENTS': 2, 'TREATS': 2, 'AUGMENTS': 2,
        'INTERACTS_WITH': 1, 'ISA': 1, 'disease_has_finding': 1, 'MANIFESTATION_OF': 1
    }

    results = []
    try:
        result = session.run(query, start_cui=start_cui, end_cui=end_cui)
        for record in result:
            if record:
                score = sum(relation_scores.get(rel, 1) for rel in record['relationships'])
                score *= 1.0 / (1 + record['total_length'])
                results.append({
                    'node_cuis': record['node_cuis'],
                    'relationships': record['relationships'],
                    'path_length': record['total_length'],
                    'score': score
                })

        results.sort(key=lambda x: x['score'], reverse=True)

    except Exception as e:
        logging.error(f"Error in enhanced path finding: {str(e)}")

    return results


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
    LIMIT 3
    """

    with self.driver.session(database='knowledge') as session:
        result = session.run(query, start_cui=start_cui, end_cui=end_cui)
        paths = []

        for record in result:
            paths.append({
                'node_names': record['node_names'],
                'relationships': record['relationships'],
                'path_length': record['path_length']
            })

        # 直接返回最短的3条路径
        return paths


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

    def _set_shortest_path_question_option(self, question: MedicalQuestion, key, question_cuis: set[str],
                                           option_cuis: set[str], database) -> None:
        try:
            with self.driver.session(database=database) as session:
                for start_cui in question_cuis:
                    for end_cui in option_cuis:
                        if start_cui == end_cui:
                            continue
                        path = _find_shortest_path_score(session, start_cui, end_cui)[:1]
                        if path:
                            question.casual_paths.no.get(key).append(
                                self.entity_processor.batch_get_names(path['node_cuis'], True))
                            question.casual_relationships.get(key).append(path['relationships'])
        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}", exc_info=True)

        question.generate_paths()  # Update path strings

    def close(self):
        """Close database connection"""
        if hasattr(self, 'driver'):
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def process_causal_graph_paths(self, start_cui: str, end_cui: str) -> List[Dict]:
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
            return list(unique_paths.values())

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
        LIMIT 3
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
            return paths

    def process_all_entity_pairs_enhance(self, question: MedicalQuestion) -> None:
        """Enhanced version of process_entity_pairs that queries and processes paths."""

        def process_wrong_pairs(entity_pairs, graph, is_causal_graph):
            seen_paths = set()
            for start_name in entity_pairs['start']:
                for end_name in entity_pairs['end']:
                    # Convert names to CUIs
                    start_cui = self.entity_processor.get_name_cui(start_name)
                    end_cui = self.entity_processor.get_name_cui(end_name)
                    if not start_cui or not end_cui or start_cui == end_cui:
                        continue

                    # 根据图类型查询并处理路径
                    if is_causal_graph:
                        paths = self.process_causal_graph_paths(start_cui, end_cui)
                    else:
                        paths = self.process_knowledge_graph_paths(start_cui, end_cui)

                    for path in paths:
                        # 创建用于去重的路径标识
                        path_identifier = (
                            tuple(path['node_names']),
                            tuple(path['relationships'])
                        )

                        # 如果是新路径，添加到结果中
                        if path_identifier not in seen_paths:
                            seen_paths.add(path_identifier)
                            graph.nodes.append(path['node_names'])
                            graph.relationships.append(path['relationships'])

        def process_aligned_pairs(entity_pairs, graph, is_causal_graph):
            seen_paths = set()
            for start_name, end_name in zip(entity_pairs['start'], entity_pairs['end']):
                # Convert names to CUIs
                start_cui = self.entity_processor.get_name_cui(start_name)
                end_cui = self.entity_processor.get_name_cui(end_name)
                if not start_cui or not end_cui or start_cui == end_cui:
                    continue

                # 根据图类型查询并处理路径
                if is_causal_graph:
                    paths = self.process_causal_graph_paths(start_cui, end_cui)
                else:
                    paths = self.process_knowledge_graph_paths(start_cui, end_cui)

                for path in paths:
                    # 创建用于去重的路径标识
                    path_identifier = (
                        tuple(path['node_names']),
                        tuple(path['relationships'])
                    )

                    # 如果是新路径，添加到结果中
                    if path_identifier not in seen_paths:
                        seen_paths.add(path_identifier)
                        graph.nodes.append(path['node_names'])
                        graph.relationships.append(path['relationships'])

        try:
            if len(question.knowledge_graph.entities_pairs['start']) == len(
                    question.knowledge_graph.entities_pairs['end']):
                process_aligned_pairs(question.knowledge_graph.entities_pairs, question.knowledge_graph,
                                      is_causal_graph=False)
            else:
                process_wrong_pairs(question.knowledge_graph.entities_pairs, question.knowledge_graph,
                                      is_causal_graph=False)
        except Exception as e:
            self.logger.error(f"Error processing knowledge graph entity pairs: {str(e)}", exc_info=True)

        try:
            if len(question.causal_graph.entities_pairs['start']) == len(
                    question.causal_graph.entities_pairs['end']):
                process_aligned_pairs(question.causal_graph.entities_pairs, question.causal_graph,
                                      is_causal_graph=False)
            else:
                process_wrong_pairs(question.causal_graph.entities_pairs, question.causal_graph,
                                    is_causal_graph=False)
        except Exception as e:
            self.logger.error(f"Error processing causal graph entity pairs: {str(e)}", exc_info=True)

        question.generate_paths()

    def process_casual_paths_enhance(self, question: MedicalQuestion, control) -> None:
        """Enhanced version of process_casual_paths that returns multiple shorter paths"""
        if control:
            return
        try:
            with self.driver.session(database='casual') as session:
                # 获取问题中的CUIs
                question_cuis = set(self.entity_processor.process_text(question.question))
                keys = list(question.options.keys())

                # 分别获取每个选项的CUIs
                options_cuis = {
                    key: set(self.entity_processor.process_text(question.options.get(key)))
                    for key in keys
                }

                # 用于去重的集合
                seen_paths = set()

                # 查找路径
                for key in keys:
                    if key not in question.casual_nodes:
                        question.casual_nodes[key] = []
                    if key not in question.casual_relationships:
                        question.casual_relationships[key] = []

                    for start_cui in question_cuis:
                        for end_cui in options_cuis[key]:
                            if start_cui == end_cui:
                                continue

                            paths = _find_shortest_path_enhance(session, start_cui, end_cui)

                            for path in paths:
                                # 创建用于去重的路径标识
                                path_identifier = (
                                    tuple(path['node_cuis']),
                                    tuple(path['relationships'])
                                )

                                # 如果是新路径，添加到结果中
                                if path_identifier not in seen_paths:
                                    seen_paths.add(path_identifier)
                                    question.casual_nodes[key].append(
                                        self.entity_processor.batch_get_names(path['node_cuis'], True)
                                    )
                                    question.casual_relationships[key].append(path['relationships'])

        except Exception as e:
            self.logger.error(f"Error in enhanced casual paths processing: {str(e)}", exc_info=True)

        question.generate_paths()  # Update path strings


def main():
    var = {
        "causal_analysis": {
            "start": ["vWF", "vWF", "Endothelial cells"],
            "end": ["Blood coagulation", "Platelet adhesion", "vWF"]
        },
        "additional_entity_pairs": {
            "start": ["Endothelial cells", "Endothelial cells", "Weibel-Palade bodies"],
            "end": ["Weibel-Palade bodies", "Blood vessel", "vWF"]
        }
    }
    question = MedicalQuestion(
        question="Sugar restricted to diet was beneficial in presence of unfavorable hygiene was from which study?",
        options={
            "opa": "Renin",
            "opb": "Angiotensin Converting Enzyme",
            "opc": "Chymase",
            "opd": "Carboxypeptidase"}, topic_name='null', is_multi_choice=True, correct_answer='opa')
    processor = QueryProcessor()
    question.causal_graph.entities_pairs = {
        "start": ["vWF", "vWF", "Endothelial cells"],
        "end": ["Blood coagulation", "Platelet adhesion", "vWF"]}
    question.knowledge_graph.entities_pairs = {
        "start": ["Endothelial cells", "Endothelial cells", "Weibel-Palade bodies"],
        "end": ["Weibel-Palade bodies", "Blood vessel", "vWF"]}
    processor.entity_processor.threshold = 0.25
    processor.process_all_entity_pairs_enhance(question)
    print(question.causal_graph.paths)
    print(question.knowledge_graph.paths)
    var = ['(Nitrates)-INHIBITS->(ethanol)-STIMULATES->(Coronary vasodilator (product))',
           '(Nitrates)-INHIBITS->(nitric oxide)-STIMULATES->(Coronary vasodilator (product))',
           '(Nitrates)-STIMULATES->(nitric oxide)-STIMULATES->(Coronary vasodilator (product))',
           '(Nitrates)-CAUSES->(Oxidative Stress)-AFFECTS->(Exocytosis)-AFFECTS->(Reduced)',
           '(Nitrates)-AFFECTS->(Cell Communication)-CAUSES->(Exocytosis)-AFFECTS->(Reduced)',
           '(Nitrates)-STIMULATES->(calcium)-CAUSES->(Exocytosis)-AFFECTS->(Reduced)']
    var = ['(Nitrates)-INTERACTS_WITH->(Coronary vasodilator (product))',
           '(Nitrates)-STIMULATES->(iron)-CAUSES->(Exocytosis)-AFFECTS->(Reduced)',
           '(Nitrates)-INHIBITS->(ascorbic acid)-CAUSES->(Exocytosis)-AFFECTS->(Reduced)',
           '(Nitrates)-STIMULATES->(ascorbic acid)-CAUSES->(Exocytosis)-AFFECTS->(Reduced)']


if __name__ == '__main__':
    """ processor = QueryProcessor()
    processor1 = EntityProcessor()
    start = processor1.get_name_cui('Trismus')
    end = processor1.get_name_cui('Inflammation')
    with processor.driver.session(database='Causal') as session:
        print(process_paths(_find_shortest_paths(session,start,end)))"""
    main()
