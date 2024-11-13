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
        self.database = db_config["database"]
        self.entity_processor = EntityProcessor()  # Still needed for name-to-CUI conversion

    def process_entity_pairs(self, question: MedicalQuestion) -> None:
        """Process entity pairs to find paths between them"""
        try:
            with self.driver.session(database=self.database) as session:
                KG_nodes = []
                KG_relationships = []
                for start_name,end_name in zip(question.entities_original_pairs['start'],question.entities_original_pairs['end']):
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
                    print(path)
                    if path:
                        KG_nodes.append(self.entity_processor.batch_get_names(path['node_cuis'],True))
                        KG_relationships.append(path['relationships'])
        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}", exc_info=True)

        question.KG_nodes.extend(KG_nodes)
        question.KG_relationships.extend(KG_relationships)
        question.generate_paths()  # Update path strings

    def process_casual_paths(self, question: MedicalQuestion) -> None:
        """Find supporting KG paths for casual relationships"""
        try:
            with self.driver.session(database=self.database) as session:
                question_cuis = self.entity_processor.process_text(question.question)
                options_cuis = []
                for option in question.options:
                    options_cuis.extend(self.entity_processor.process_text(option))

                # 存储所有找到的路径
                casual_nodes = []
                casual_relationships = []

                # Find paths between consecutive nodes
                for start in question_cuis:
                    for end in options_cuis:
                        path = _find_shortest_path(
                            session,
                            start,
                            end
                        )
                        if path:
                            casual_nodes.append(self.entity_processor.batch_get_names(path['node_cuis'],False))
                            casual_relationships.append(path['relationships'])

        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}", exc_info=True)
        question.casual_nodes.extend(casual_nodes)
        question.casual_relationships.extend(casual_relationships)
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
    processor = QueryProcessor()
    start = "C0027750"
    end = "C4330852"
    print(_find_shortest_path(processor.driver.session(database='knowledge'), start, end))