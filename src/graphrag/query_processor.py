from neo4j import GraphDatabase
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from config import config
from src.modules.data_format import DataFormat, OptionKey, EntityPairs
from src.graphrag.entity_processor import EntityProcessor

class QueryProcessor:
    """Knowledge graph query processor focused on medical QA path finding"""

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
        self.entity_processor = EntityProcessor()

    def query_question_cuis(self, cuis: 'DataFormat.QuestionCUIs') -> 'DataFormat.QuestionGraphResult':
        """
        Find paths for question CUIs and option CUIs

        Args:
            cuis: QuestionCUIs containing question and option CUIs

        Returns:
            QuestionGraphResult containing paths between entities
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Process question internal paths
                question_paths = self._find_internal_paths(
                    session,
                    cuis.question_cuis,
                    cuis.question_entities
                )

                # Process paths to each option
                option_paths = {}
                for option_key, option_cuis in cuis.option_cuis.items():
                    paths = self._find_paths_between(
                        session,
                        cuis.question_cuis,
                        option_cuis,
                        cuis.question_entities,
                        cuis.option_entities[option_key]
                    )
                    option_paths[option_key] = paths

                return DataFormat.QuestionGraphResult(
                    question_paths=question_paths,
                    option_paths=option_paths
                )

        except Exception as e:
            self.logger.error(f"Error in query_question_cuis: {str(e)}")
            return DataFormat.QuestionGraphResult(
                question_paths=[],
                option_paths={k: [] for k in OptionKey}
            )

    def query_entity_pairs(self, entity_names: EntityPairs) -> List[DataFormat.GraphPath]:
        """
        Find shortest path between specified entity pairs

        Args:
            entity_pairs: EntityPairs object containing lists of start and end entities

        Returns:
            List of GraphPath objects showing shortest connections between entities
        """
        try:
            entity_pairs = self.entity_processor.process_entity_pairs(entity_names)
            print(entity_pairs)
            paths = []

            with self.driver.session(database=self.database) as session:
                # Try each combination once
                for start_cui in entity_pairs.start:
                    for end_cui in entity_pairs.end:
                        query = """
                        MATCH path = shortestPath((start:Entity)-[*..3]-(end:Entity))
                        WHERE start.cui = $start_cui AND end.cui = $end_cui
                        RETURN 
                            [node in nodes(path) | node.cui] as node_cuis,
                            [node in nodes(path) | node.name] as node_names,
                            [rel in relationships(path) | rel.name] as relationships
                        LIMIT 1
                        """

                        self.logger.debug(f"Searching path between CUIs '{start_cui}' and '{end_cui}'")

                        # Execute query with CUIs
                        result = session.run(
                            query,
                            start_cui=start_cui,
                            end_cui=end_cui
                        )

                        # Process results
                        record = result.single()
                        if record:
                            self.logger.debug(f"Found path: {record}")
                            node_cuis = record["node_cuis"]
                            path = DataFormat.GraphPath(
                                nodes=node_cuis,
                                node_names=[self.entity_processor.get_entity_info(cui).name for cui in node_cuis],
                                relationships=record["relationships"],
                                source_entity=self.entity_processor.get_entity_info(node_cuis[0]),
                                target_entity=self.entity_processor.get_entity_info(node_cuis[-1])
                            )
                            paths.append(path)
                            break  # Found a path, no need to check other combinations

                if not paths:
                    self.logger.warning(
                        f"No paths found between entities. Start: {entity_pairs.start}, End: {entity_pairs.end}"
                    )
                else:
                    self.logger.info(f"Found path between entities")

                return paths

        except Exception as e:
            self.logger.error(f"Error in query_entity_pairs: {str(e)}", exc_info=True)
            return []

    def _find_internal_paths(self, session, cuis: List[str],
                             entities: List['DataFormat.EntityInfo']) -> List['DataFormat.GraphPath']:
        """Find paths between entities within the question"""
        query = """
        MATCH (start:Entity)
        WHERE start.cui IN $cuis
        MATCH (end:Entity)
        WHERE end.cui IN $cuis AND start <> end
        MATCH path = (start)-[*..3]->(end)
        RETURN 
            [node in nodes(path) | node.cui] as cuis,
            [node in nodes(path) | node.name] as names,
            [rel in relationships(path) | rel.name] as rels
        ORDER BY length(path)
        LIMIT 5
        """

        result = session.run(query, cuis=cuis)
        paths = []
        for record in result:
            # Get entity info for each node in the path
            node_cuis = record["cuis"]
            path_entities = [self.entity_processor.get_entity_info(cui) for cui in node_cuis]

            # Create path with entity info
            path = self._create_graph_path(record, path_entities)
            paths.append(path)
        return paths

    def _find_paths_between(self, session, start_cuis: List[str], end_cuis: List[str],
                            start_entities: List['DataFormat.EntityInfo'],
                            end_entities: List['DataFormat.EntityInfo']) -> List['DataFormat.GraphPath']:
        """Find paths between two sets of CUIs"""
        query = """
        MATCH (start:Entity)
        WHERE start.cui IN $start_cuis
        MATCH (end:Entity)
        WHERE end.cui IN $end_cuis
        MATCH path = (start)-[*..3]->(end)
        WHERE NOT start = end
        RETURN 
            [node in nodes(path) | node.cui] as cuis,
            [node in nodes(path) | node.name] as names,
            [rel in relationships(path) | rel.name ] as rels
        ORDER BY length(path)
        LIMIT 5
        """

        result = session.run(query, start_cuis=start_cuis, end_cuis=end_cuis)
        paths = []
        for record in result:
            # Get entity info for each node in the path
            node_cuis = record["cuis"]
            path_entities = [self.entity_processor.get_entity_info(cui) for cui in node_cuis]

            # Create path with entity info
            path = self._create_graph_path(record, path_entities)
            paths.append(path)
        return paths

    def _create_graph_path(self, record: Dict,
                           available_entities: List['DataFormat.EntityInfo']) -> 'DataFormat.GraphPath':
        """Create a GraphPath object from a query record"""
        cuis = record["cuis"]

        # Use provided entities or get from processor if not available
        node_entities = []
        for i, cui in enumerate(cuis):
            entity = next(
                (e for e in available_entities if e.cui == cui),
                None
            )
            node_entities.append(entity if entity else self.entity_processor.get_entity_info(cui))

        return DataFormat.GraphPath(
            nodes=cuis,
            node_names=[e.name if e else cui for e, cui in zip(node_entities, cuis)],
            relationships=record["rels"],
            source_entity=node_entities[0],
            target_entity=node_entities[-1]
        )

    def _find_entity_pair_paths(self, session, pair: 'DataFormat.EntityPairs') -> List['DataFormat.GraphPath']:
        """Find paths between a specific entity pair"""
        query = """
        MATCH (start:Entity {name: $start_name})
        MATCH (end:Entity {name: $end_name})
        MATCH path = (start)-[*..3]->(end)
        WHERE NOT start = end
        RETURN 
            [node in nodes(path) | node.cui] as cuis,
            [node in nodes(path) | node.name] as names,
            [rel in relationships(path) | rel.name] as rels
        ORDER BY length(path)
        LIMIT 5
        """

        result = session.run(query,
                             start_name=pair.start,
                             end_name=pair.end)
        return [
            self._create_graph_path(record, [])
            for record in result
        ]

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
    data = DataFormat.EntityPairs(
        start=["saltatory conduction", "saltatory transmission"],
        end=["impulse conduction", "nerve impulse propagation"],
        reasoning="Examining the process of saltatory conduction will help verify the correctness of option c."
    )
    print(processor.query_entity_pairs(data))