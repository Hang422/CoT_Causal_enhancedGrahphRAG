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

                    path = _find_shortest_path(
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
                keys =['opa', 'opb', 'opc', 'opd']
                # 分别获取每个选项的CUIs
                options_cuis_a = set(self.entity_processor.process_text(question.options.get('opa')))
                options_cuis_b = set(self.entity_processor.process_text(question.options.get('opb')))
                options_cuis_c = set(self.entity_processor.process_text(question.options.get('opc')))
                options_cuis_d = set(self.entity_processor.process_text(question.options.get('opd')))
                cuis = [options_cuis_a, options_cuis_b, options_cuis_c, options_cuis_d]

                # 查找路径
                for option_cuis, key in zip(cuis,keys):
                    self._set_shortest_path_question_option(question,key,question_cuis, option_cuis, database)

        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}", exc_info=True)

        question.generate_paths()  # Update path strings

    def _set_shortest_path_question_option(self, question: MedicalQuestion, key, question_cuis: set[str], option_cuis:set[str], database) -> None:
        try:
            with self.driver.session(database=database) as session:
                for start_cui in question_cuis:
                    for end_cui in option_cuis:
                        if start_cui == end_cui:
                            continue
                        path = _find_shortest_path(session, start_cui, end_cui)
                        if path:
                            question.casual_nodes.get(key).append(self.entity_processor.batch_get_names(path['node_cuis'], True))
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


if __name__ == '__main__':
    question = MedicalQuestion("Heavy forces on periodontal ligament causes:", {
        "opa": "Hyalinization",
        "opb": "Osteoclastic activity around tooth",
        "opc": "Osteoblastic activity around tooth",
        "opd": "Crest bone resorption"})
    processor = QueryProcessor()
    question.entities_original_pairs = {
        "start": [
            "Dorsal respiratory group",
            "Ventral respiratory neurons"
        ],
        "end": [
            "Pre-Botzinger complex",
            "Pneumotaxic center"
        ]}
    processor.process_casual_paths(question, 'knowledge')
    print(question.casual_paths)
