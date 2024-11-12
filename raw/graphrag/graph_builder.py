from neo4j import GraphDatabase
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from tqdm import tqdm
import psutil
import os


class KnowledgeGraphBuilder:
    def __init__(self, uri: str, username: str, password: str, database: str):
        """
        Initialize connection to Neo4j database
        """
        self.memory = psutil.virtual_memory()
        self.cpu_count = os.cpu_count()

        # 优化参数
        self.batch_size = 5000  # 更大的批处理大小，因为我们不需要存储那么多属性了
        self.max_workers = 6

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_lifetime=3600,
            max_connection_pool_size=self.max_workers * 2
        )
        self.database = database
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"System Info: {self.cpu_count} cores, "
                         f"{self.memory.total / (1024 ** 3):.1f}GB RAM, "
                         f"Batch size: {self.batch_size}")

    def close(self):
        if hasattr(self, 'driver'):
            self.driver.close()

    def create_constraints(self):
        """Create constraint for CUI"""
        with self.driver.session(database=self.database) as session:
            try:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.cui IS UNIQUE")
            except Exception as e:
                self.logger.warning(f"Error creating constraint: {str(e)}")

    def create_graph_from_batch(self, tx, batch_data: List[Dict]):
        """Create both nodes and relationships in a single batch"""
        query = """
        UNWIND $batch as row
        MERGE (e1:Entity {cui: row.entity1_cui})
        MERGE (e2:Entity {cui: row.entity2_cui})
        MERGE (e1)-[r:RELATION {name: row.relation_name}]->(e2)
        """
        return tx.run(query, batch=batch_data)

    def process_batch(self, batch_data: List[Dict]):
        """Process a single batch of data"""
        with self.driver.session(database=self.database) as session:
            session.execute_write(self.create_graph_from_batch, batch_data)

    def build_graph(self, relations_file: str):
        """Build the entire graph from relations file"""
        try:
            # 创建约束
            self.create_constraints()

            # 读取CSV文件
            self.logger.info(f"Reading relations from: {relations_file}")

            # 使用迭代器读取大文件
            chunks = pd.read_csv(
                relations_file,
                chunksize=self.batch_size,
                dtype={
                    'Entity1_CUI': str,
                    'Entity2_CUI': str,
                    'Relation': str
                }
            )

            # 处理每个数据块
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for chunk in tqdm(chunks, desc="Processing data"):
                    # 准备批处理数据
                    batch_data = [{
                        'entity1_cui': row['Entity1_CUI'],
                        'entity2_cui': row['Entity2_CUI'],
                        'relation_name': row['Relation_Name']
                    } for _, row in chunk.iterrows()]

                    # 处理批次
                    self.process_batch(batch_data)

        except Exception as e:
            self.logger.error(f"Error building graph: {str(e)}")
            raise


def main():
    # Neo4j连接参数
    neo4j_params = {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "luohang819",
        "database": "casual-1"
    }

    graph_builder = None
    try:
        # 初始化图构建器
        graph_builder = KnowledgeGraphBuilder(**neo4j_params)

        # 构建图
        graph_builder.build_graph("../data/database/Processed/semmedb/causal_relations_output.csv")

        # 验证数据
        with graph_builder.driver.session(database=graph_builder.database) as session:
            # 获取节点数量
            node_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()['count']
            # 获取关系数量
            rel_count = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count").single()['count']

            print(f"\nGraph Statistics:")
            print(f"Total Entities: {node_count}")
            print(f"Total Relations: {rel_count}")

    except Exception as e:
        logging.error(f"Error: {str(e)}")
    finally:
        if graph_builder:
            graph_builder.close()


if __name__ == "__main__":
    main()