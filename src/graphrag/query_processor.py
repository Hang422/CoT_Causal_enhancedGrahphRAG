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
    MATCH path = shortestPath((start:Entity)-[*..5]->(end:Entity))
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


def _find_shortest_path_score(session, start_cui: str, end_cui: str) -> Optional[Dict]:
    """Find highest scoring path between two CUIs"""
    query = """
   MATCH path = allShortestPaths((start:Entity)-[*..5]->(end:Entity))
   WHERE start.cui = $start_cui AND end.cui = $end_cui
   RETURN 
       [node in nodes(path) | node.cui] as node_cuis,
       [rel in relationships(path) | rel.name] as relationships,
       length(path) as path_length
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

    result = session.run(query, start_cui=start_cui, end_cui=end_cui)
    best_path = None
    best_score = -1

    for record in result:
        score = sum(relation_scores.get(rel, 1) for rel in record['relationships'])
        score *= 1.0 / (1 + record['path_length'])
        if score > best_score:
            best_score = score
            best_path = record

    return best_path


def _find_shortest_path_score_both_directions(session, start_cui: str, end_cui: str) -> Optional[Dict]:
    pass


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

    def process_entity_pairs(self, question: MedicalQuestion, database) -> None:
        """Process entity pairs to find paths between them"""
        paths = []
        try:
            with self.driver.session(database=database) as session:
                for start_name, end_name in zip(question.entities_original_pairs['start'],
                                                question.entities_original_pairs['end']):
                    # Convert names to CUIs
                    start_cui = self.entity_processor.get_name_cui(start_name)
                    end_cui = self.entity_processor.get_name_cui(end_name)
                    if not start_cui or not end_cui:
                        continue

                    if start_cui == end_cui:
                        continue

                    path = _find_shortest_path_score(
                        session,
                        start_cui,
                        end_cui
                    )
                    if path is not None and path not in paths:
                        question.KG_nodes.append(self.entity_processor.batch_get_names(path['node_cuis'], True))
                        question.KG_relationships.append(path['relationships'])
                        paths.append(path)
        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}", exc_info=True)

        question.generate_paths()  # Update path strings

    def process_casual_paths(self, question: MedicalQuestion, database) -> None:
        """Find supporting KG paths for casual relationships"""
        try:
            with self.driver.session(database=database) as session:
                # 获取问题中的CUIs
                question_cuis = set(self.entity_processor.process_text(question.question))
                keys = ['opa', 'opb', 'opc', 'opd']
                # 分别获取每个选项的CUIs
                options_cuis_a = set(self.entity_processor.process_text(question.options.get('opa')))
                options_cuis_b = set(self.entity_processor.process_text(question.options.get('opb')))
                options_cuis_c = set(self.entity_processor.process_text(question.options.get('opc')))
                options_cuis_d = set(self.entity_processor.process_text(question.options.get('opd')))
                cuis = [options_cuis_a, options_cuis_b, options_cuis_c, options_cuis_d]

                # 查找路径
                for option_cuis, key in zip(cuis, keys):
                    self._set_shortest_path_question_option(question, key, question_cuis, option_cuis, database)

        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}", exc_info=True)

        question.generate_paths()  # Update path strings

    def _set_shortest_path_question_option(self, question: MedicalQuestion, key, question_cuis: set[str],
                                           option_cuis: set[str], database) -> None:
        try:
            with self.driver.session(database=database) as session:
                for start_cui in question_cuis:
                    for end_cui in option_cuis:
                        if start_cui == end_cui:
                            continue
                        path = _find_shortest_path_score(session, start_cui, end_cui)
                        if path:
                            question.casual_nodes.get(key).append(
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

    def process_entity_pairs_enhance(self, question: MedicalQuestion, database) -> None:
        """Enhanced version of process_entity_pairs that finds multiple shorter paths"""

        # 用于去重的集合
        seen_paths = set()

        try:
            with self.driver.session(database=database) as session:
                for start_name, end_name in zip(question.entities_original_pairs['start'],
                                                question.entities_original_pairs['end']):
                    # Convert names to CUIs
                    start_cui = self.entity_processor.get_name_cui(start_name)
                    end_cui = self.entity_processor.get_name_cui(end_name)

                    if not start_cui or not end_cui:
                        continue

                    if start_cui == end_cui:
                        continue

                    # 使用增强版路径查找
                    paths = _find_shortest_path_enhance(
                        session,
                        start_cui,
                        end_cui
                    )

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
                            question.KG_nodes.append(
                                self.entity_processor.batch_get_names(path['node_cuis'], True)
                            )
                            question.KG_relationships.append(path['relationships'])

        except Exception as e:
            self.logger.error(f"Error in enhanced entity pairs processing: {str(e)}", exc_info=True)

        question.generate_paths()  # Update path strings


def _find_shortest_path_enhance(session, start_cui: str, end_cui: str) -> List[Dict]:
    """Enhanced path finding that returns alternative shorter paths"""
    query = """
    // 获取原始最短路径和长度
    MATCH (start:Entity {cui: $start_cui}), (end:Entity {cui: $end_cui})
    OPTIONAL MATCH initialPath = shortestPath((start)-[*..4]->(end))
    WITH start, end, 
         CASE WHEN initialPath IS NULL THEN -1 ELSE length(initialPath) END as original_length
    WHERE original_length > 0

    // 获取邻居并寻找更短的路径
    MATCH (start)-[r1]-(neighbor)
    WHERE neighbor <> end
    OPTIONAL MATCH alterPath = shortestPath((neighbor)-[*..3]->(end))
    WITH start, neighbor, alterPath, original_length,
         CASE WHEN alterPath IS NULL THEN -1 ELSE length(alterPath) END as path_length
    WHERE alterPath IS NOT NULL AND path_length < original_length

    // 返回完整路径信息
    RETURN DISTINCT
           [node in nodes(alterPath) | node.cui] as node_cuis,
           [rel in relationships(alterPath) | rel.name] as relationships,
           path_length
    ORDER BY path_length ASC
    LIMIT 5
    """

    results = []
    try:
        result = session.run(query, start_cui=start_cui, end_cui=end_cui)
        for record in result:
            if record:
                results.append({
                    'node_cuis': record['node_cuis'],
                    'relationships': record['relationships'],
                    'path_length': record['path_length']
                })
    except Exception as e:
        logging.error(f"Error in enhanced path finding: {str(e)}")

    return results

def process_casual_paths_enhance(self, question: MedicalQuestion, database) -> None:
    """Enhanced version of process_casual_paths that returns multiple shorter paths"""
    try:
        with self.driver.session(database=database) as session:
            # 获取问题中的CUIs
            question_cuis = set(self.entity_processor.process_text(question.question))
            keys = ['opa', 'opb', 'opc', 'opd']
            # 分别获取每个选项的CUIs
            options_cuis = {
                'opa': set(self.entity_processor.process_text(question.options.get('opa'))),
                'opb': set(self.entity_processor.process_text(question.options.get('opb'))),
                'opc': set(self.entity_processor.process_text(question.options.get('opc'))),
                'opd': set(self.entity_processor.process_text(question.options.get('opd')))
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
    question = MedicalQuestion("Heavy forces on periodontal ligament causes:", {
        "opa": "Hyalinization",
        "opb": "Osteoclastic activity around tooth",
        "opc": "Osteoblastic activity around tooth",
        "opd": "Crest bone resorption"})
    processor = QueryProcessor()
    question.entities_original_pairs = {
        "start": [
            "lactase deficiency",
            "yogurt",
            "condensed milk"
        ],
        "end": [
            "lactose",
            "live cultures",
            "lactose content"
        ]}
    processor.process_entity_pairs_enhance(question, 'casual')
    print(question.KG_paths)
