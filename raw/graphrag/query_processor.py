import json
from neo4j import GraphDatabase
from typing import List, Dict, Optional
from raw.graphrag.entity_processor import MedicalCUIExtractor
from tqdm import tqdm

class EntityPathFinder:
    def __init__(self, neo4j_params: Dict[str, str]):
        """初始化Neo4j连接和CUI转换器

        Args:
            neo4j_params: 包含Neo4j连接参数的字典 (uri, username, password, database)
        """
        self.driver = GraphDatabase.driver(
            neo4j_params["uri"],
            auth=(neo4j_params["username"], neo4j_params["password"])
        )
        self.database = neo4j_params["database"]
        self.extractor = MedicalCUIExtractor()

    def close(self):
        """关闭Neo4j连接"""
        self.driver.close()

    def find_paths(self,
                   question_cuis: List[str],
                   option_cuis: List[str],
                   max_depth: int = 2) -> List[str]:
        """查找问题实体和选项实体之间的最短路径

        Args:
            question_cuis: 问题中的CUI列表
            option_cuis: 选项中的CUI列表
            max_depth: 最大路径深度

        Returns:
            List[str]: 包含最短路径的列表（如果存在多条相同长度的最短路径，则全部返回）
        """
        paths_with_length = []  # 存储路径及其长度

        with self.driver.session(database=self.database) as session:
            # 修改查询以获取路径长度并按长度排序
            query = """
            MATCH (start)
            WHERE start.cui IN $question_cuis
            MATCH (end)
            WHERE end.cui IN $option_cuis
            MATCH path = shortestPath((start)-[*..%d]-(end))
            WHERE start <> end
            RETURN 
                [node in nodes(path) | node.cui] as path_cuis,
                [rel in relationships(path) | rel.name] as relationships,
                length(path) as path_length
            ORDER BY path_length ASC
            """ % max_depth

            result = session.run(
                query,
                question_cuis=question_cuis,
                option_cuis=option_cuis
            )

            min_length = float('inf')  # 初始化最小长度
            shortest_paths = []  # 存储最短路径

            for record in result:
                try:
                    current_length = record["path_length"]

                    # 如果找到了更短的路径，清空之前的路径
                    if current_length < min_length:
                        min_length = current_length
                        shortest_paths = []

                    # 如果不是最短路径，跳过
                    if current_length > min_length:
                        continue

                    # 构建路径字符串
                    node_names = []
                    for cui in record["path_cuis"]:
                        name = self.extractor.get_cui_name(cui)
                        node_names.append(name if name else cui)

                    relationships = record["relationships"]

                    path_parts = []
                    for i in range(len(node_names)):
                        path_parts.append(node_names[i])
                        if i < len(relationships):
                            path_parts.append(f"[{relationships[i]}]")

                    path_str = " -> ".join(path_parts)
                    shortest_paths.append(path_str)

                except Exception as e:
                    print(f"Error processing path: {str(e)}")
                    continue

            return shortest_paths

    def get_shortest_path_length(self, cui1: str, cui2: str, directed: bool = True) -> Optional[int]:
        """
        获取两个CUI之间的最短路径长度

        Args:
            cui1: 起始CUI
            cui2: 目标CUI
            directed: 是否只考虑有向路径（默认True）

        Returns:
            int: 最短路径长度，如果不存在路径则返回None
        """
        with self.driver.session(database=self.database) as session:
            if directed:
                # 只考虑有向路径
                query = """
                   MATCH (start:Entity {cui: $cui1}), (end:Entity {cui: $cui2})
                   MATCH path = shortestPath((start)-[*]->(end))
                   RETURN length(path) as path_length
                   """
            else:
                # 考虑无向路径
                query = """
                   MATCH (start:Entity {cui: $cui1}), (end:Entity {cui: $cui2})
                   MATCH path = shortestPath((start)-[*]-(end))
                   RETURN length(path) as path_length
                   """

            result = session.run(query, cui1=cui1, cui2=cui2)
            record = result.single()

            return record["path_length"] if record else None


if __name__ == "__main__":
    # Neo4j连接参数
    neo4j_params = {
        "uri": "bolt://localhost:7687",
        "username": "neo4j" ,
        "password": "luohang819",
        "database": "casual"
    }

    # 初始化PathFinder
    finder = EntityPathFinder(neo4j_params)

    try:
        # 读取包含CUI的实体对数据
        with open('../temp/pairs_with_entity.json', 'r', encoding='utf-8') as f:
            questions = json.load(f)

        # 处理每个问题
        processed_questions = []
        for question in tqdm(questions, desc="Processing questions"):
            entity_pairs = question['analysis_result']['entity_pairs']

            # 为每个实体对找路径
            paths_results = []
            for pair in entity_pairs:
                start_cui = pair['start']['cui']
                end_cui = pair['end']['cui']

                if start_cui and end_cui:
                    paths = finder.find_paths([start_cui], [end_cui])
                    paths_results.append({
                        'start': pair['start'],
                        'end': pair['end'],
                        'paths': paths
                    })

            # 组织问题的完整信息
            processed_question = {
                'question': question['question'],
                'options': question['options'],
                'reasoning_chain': question['analysis_result']['reasoning_chain'],
                'entity_pairs_paths': paths_results
            }
            processed_questions.append(processed_question)

        # 保存结果
        with open('../temp/query_results.json', 'w', encoding='utf-8') as f:
            json.dump(processed_questions, f, indent=2, ensure_ascii=False)

    finally:
        finder.close()