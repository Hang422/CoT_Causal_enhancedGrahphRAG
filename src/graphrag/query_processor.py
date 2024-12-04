from neo4j import GraphDatabase
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from config import config
from src.modules.MedicalQuestion import MedicalQuestion
from src.graphrag.entity_processor import EntityProcessor


def _find_shortest_path(session, start_cui: str, end_cui: str) -> Optional[Dict]:
    """Find shortest path between two CUIs"""
    query = """
    MATCH path = shortestPath((start)-[*..5]->(end))
    WHERE start.cui = $start_cui AND end.cui = $end_cui
    RETURN 
        [node in nodes(path) | node.cui] as node_cuis,
        [rel in relationships(path) | rel.name] as relationships
    LIMIT 1
    """

    result = session.run(
        query,
        start_cui=start_cui,
        end_cui=end_cui
    )
    record = result.single()
    return record if record else None


def _find_shortest_path_score(session, start_cui: str, end_cui: str) -> List[Dict]:
    """Find highest scoring paths between two CUIs with length constraints"""
    query = """
    MATCH path = allShortestPaths((start)-[*..5]->(end))
    WHERE start.cui = $start_cui AND end.cui = $end_cui
    WITH path, length(path) as len
    ORDER BY len ASC
    WITH collect({
        nodes: [node in nodes(path) | node.cui],
        rels: [rel in relationships(path) | rel.name],
        length: len
    }) as paths,
    min(len) as shortest_len
    UNWIND [p in paths WHERE p.length <= shortest_len + 1] as filtered_path
    RETURN 
        filtered_path.nodes as node_cuis,
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
            'node_cuis': record['node_cuis'],
            'relationships': record['relationships'],
            'path_length': record['path_length'],
            'score': score
        })

    scored_paths.sort(key=lambda x: x['score'], reverse=True)
    return scored_paths[:3]


def _find_shortest_path_score_both_directions(session, start_cui: str, end_cui: str) -> Optional[Dict]:
    pass


def _find_shortest_path_enhance(session, start_cui: str, end_cui: str) -> List[Dict]:
    """Enhanced path finding that returns alternative shorter paths with scoring"""
    query = """
    // Get original shortest path and length
    MATCH (start {cui: $start_cui}), (end {cui: $end_cui})
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

    def process_all_entity_pairs_enhance(self, question: MedicalQuestion) -> None:
        """Enhanced version of process_entity_pairs that finds multiple shorter paths"""

        try:
            seen_paths = set()
            with self.driver.session(database='knowledge') as session:
                for start_name, end_name in zip(question.knowledge_graph.entities_pairs['start'],
                                                question.knowledge_graph.entities_pairs['end']):
                    # Convert names to CUIs
                    start_cui = self.entity_processor.get_name_cui(start_name)
                    end_cui = self.entity_processor.get_name_cui(end_name)
                    if not start_cui or not end_cui:
                        continue

                    if start_cui == end_cui:
                        continue

                    # 使用增强版路径查找
                    paths = _find_shortest_path_score(
                        session,
                        start_cui,
                        end_cui
                    )[:2]

                    # 处理找到的多条路径
                    for path in paths:
                        # 创建用于去重的路径标识
                        path_identifier = (
                            tuple(path['node_cuis']),
                            tuple(path['relationships'])
                        )

                        # 如果是新路径，添加到结果中
                        if path_identifier not in seen_paths:
                            seen_paths.add(path_identifier)
                            question.knowledge_graph.nodes.append(
                                self.entity_processor.batch_get_names(path['node_cuis'], True)
                            )
                            question.knowledge_graph.relationships.append(path['relationships'])

        except Exception as e:
            self.logger.error(f"Error in enhanced entity pairs processing: {str(e)}", exc_info=True)

        try:
            seen_paths = set()
            with self.driver.session(database='casual') as session:
                for start_name, end_name in zip(question.causal_graph.entities_pairs['start'],
                                                question.causal_graph.entities_pairs['end']):
                    # Convert names to CUIs
                    start_cui = self.entity_processor.get_name_cui(start_name)
                    end_cui = self.entity_processor.get_name_cui(end_name)
                    if not start_cui or not end_cui:
                        continue

                    if start_cui == end_cui:
                        continue

                        # 使用增强版路径查找
                    paths = _find_shortest_path_score(
                        session,
                        start_cui,
                        end_cui
                    )[:2]

                    # 处理找到的多条路径
                    for path in paths:
                        # 创建用于去重的路径标识
                        path_identifier = (
                            tuple(path['node_cuis']),
                            tuple(path['relationships'])
                        )

                        # 如果是新路径，添加到结果中
                        if path_identifier not in seen_paths:
                            seen_paths.add(path_identifier)
                            question.causal_graph.nodes.append(
                                self.entity_processor.batch_get_names(path['node_cuis'], True)
                            )
                            question.causal_graph.relationships.append(path['relationships'])

        except Exception as e:
            self.logger.error(f"Error in enhanced entity pairs processing: {str(e)}", exc_info=True)

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


if __name__ == '__main__':
    question = MedicalQuestion(
        question="Sugar restricted to diet was beneficial in presence of unfavorable hygiene was from which study?",
        options={
            "opa": "Hopewood",
            "opb": "Experimental",
            "opc": "Vipeholm",
            "opd": "Turku"}, topic_name='null', is_multi_choice=True, correct_answer='opa')
    processor = QueryProcessor()
    question.causal_graph.entities_pairs = {
        "start": [
            "Nitrates",
            "Nitrates"
        ],
        "end": [
            "Coronary vasodilator",
            "Decrease in preload"
        ]}
    question.knowledge_graph.entities_pairs = {"start": [
        "Nitrates",
        "Venodilation"
    ],
        "end": [
            "Coronary vasodilator",
            "Decrease in preload"
        ]}
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
