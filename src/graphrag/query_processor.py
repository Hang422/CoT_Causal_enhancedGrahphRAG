from neo4j import GraphDatabase
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from config import config
from src.modules.MedicalQuestion import MedicalQuestion
from src.graphrag.entity_processor import EntityProcessor

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

    def close(self):
        """Close database connection"""
        if hasattr(self, 'driver'):
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def process_causal_graph_paths_normal(self,start_cui:str, end_cui:str) -> List[Dict]:
        pass

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

            return sorted_paths[:3]

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

    def generate_initial_causal_graph(self, question: MedicalQuestion) -> None:
        """Enhanced version of process_casual_paths that returns multiple shorter paths"""
        try:
            with self.driver.session(database='causal') as session:
                # 获取问题中的CUIs
                question_cuis = set(self.entity_processor.process_text(question.question))
                keys = list(question.options.keys())

                # 分别获取每个选项的CUIs
                options_cuis = set()
                for key, option_text in question.options.items():  # 直接遍历选项字典
                    processed_cuis = self.entity_processor.process_text(option_text)
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

    def process_chain_of_thoughts(self, question: MedicalQuestion, graph:str, enhancement:bool) -> None:
        """处理思维链的路径检索"""

        for chain in question.reasoning_chain:
            # 分割思维链的步骤
            steps = chain.split('->')
            # 遍历相邻步骤对
            for i in range(len(steps) - 1):
                start = steps[i].strip()
                end = steps[i + 1].strip()
                # 通过entity processor获取CUI
                start_cuis = self.entity_processor.process_text(start)
                end_cuis = self.entity_processor.process_text(end)
                for start_cui, end_cui in zip(start_cuis, end_cuis):
                    if start_cui == end_cui:
                        continue
                    if start_cui and end_cui:
                        # 查找最短路径
                        if graph == 'both':
                            if enhancement:
                                paths = self.process_causal_graph_paths_enhanced(start_cui, end_cui)
                            else:
                                paths = self.process_causal_graph_paths_normal(start_cui, end_cui)
                            if len(paths) > 0:
                                for path in paths:
                                    question.causal_graph.nodes.append(path['node_names'])
                                    question.causal_graph.relationships.append(path['relationships'])
                            else:
                                paths = self.process_knowledge_graph_paths(start_cui, end_cui)
                                if len(paths) > 0:
                                    for path in paths:
                                        question.knowledge_graph.nodes.append(path['node_names'])
                                        question.knowledge_graph.relationships.append(path['relationships'])
                        if graph == 'knowledge':
                            paths = self.process_knowledge_graph_paths(start_cui, end_cui)
                            if len(paths) > 0:
                                for path in paths:
                                    question.knowledge_graph.nodes.append(path['node_names'])
                                    question.knowledge_graph.relationships.append(path['relationships'])
                        if graph == 'causal':
                            if enhancement:
                                paths = self.process_causal_graph_paths_enhanced(start_cui, end_cui)
                            else:
                                paths = self.process_causal_graph_paths_normal(start_cui, end_cui)
                            if len(paths) > 0:
                                for path in paths:
                                    question.causal_graph.nodes.append(path['node_names'])
                                    question.causal_graph.relationships.append(path['relationships'])

        question.generate_paths()


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
    processor.generate_initial_causal_graph(question)
    print(question.initial_causal_graph.paths)
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
    processor = QueryProcessor()
    path = config.paths["cache"] / 'test1' / 'original'
    question = MedicalQuestion.from_cache(path, "DOC for bacterial vaginosis in pregnancy")
    question.reasoning_chain = [
        'Vaginal discharge + "fishy" odor -> suggests Bacterial Vaginosis -> Treatment with Metronidazole',
        'Vaginal pH >4.5 + clue cells on microscopy -> suggests Bacterial Vaginosis -> Treatment with Clindamycin',
        'Pregnant patient with vaginal itching + thin, grayish-white discharge -> suggests Bacterial Vaginosis -> Treatment with Rovamycin']
    processor.process_chain_of_thoughts(question)
    print(question.initial_causal_graph.paths)
    print(question.causal_graph.paths)
    print(question.knowledge_graph.paths)
