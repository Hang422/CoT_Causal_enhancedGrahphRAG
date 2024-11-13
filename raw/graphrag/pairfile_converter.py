import json
from typing import Dict, Tuple
from raw.graphrag.entity_processor import MedicalCUIExtractor


class EntityCUIConverter:
    def __init__(self, threshold: float = 0.7):  # 降低默认阈值
        """
        初始化转换器

        Args:
            threshold: 实体链接的置信度阈值
        """
        self.extractor = MedicalCUIExtractor(threshold=threshold)

    def preprocess_text(self, text: str) -> str:
        """
        预处理文本以提高匹配率

        Args:
            text: 输入文本

        Returns:
            str: 预处理后的文本
        """
        # 移除常见的非关键词
        text = text.replace(" of ", " ").replace(" the ", " ")
        # 标准化空格
        text = " ".join(text.split())
        return text

    def try_alternative_matches(self, text: str) -> Tuple[str, float]:
        """
        尝试不同的文本变体来找到匹配

        Args:
            text: 原始文本

        Returns:
            Tuple[str, float]: (CUI, 置信度)
        """
        variations = [
            text,
            text.lower(),
            text.title(),
            self.preprocess_text(text),
            # 处理复合词
            " ".join(text.split(",")),
            text.split()[0] if text.split() else text  # 尝试只匹配第一个词
        ]

        best_match = None
        best_score = 0

        for variant in variations:
            results = self.extractor.get_cuis_with_context(variant)
            if results['cuis'] and results['entities']:
                score = results['entities'][0].get('confidence', 0)
                if score > best_score:
                    best_score = score
                    best_match = results['cuis'][0]

        return best_match, best_score

    def convert_entity_pairs_to_cuis(self, input_data: Dict) -> Dict:
        """
        将问题分析结果中的实体对转换为对应的CUI

        Args:
            input_data: 输入的问题数据字典

        Returns:
            Dict: 转换后的问题数据
        """
        processed_data = input_data.copy()
        entity_pairs = processed_data.get('analysis_result', {}).get('entity_pairs', [])
        processed_pairs = []

        for pair in entity_pairs:
            # 尝试多种匹配方式
            start_cui, start_score = self.try_alternative_matches(pair['start'])
            end_cui, end_score = self.try_alternative_matches(pair['end'])

            # 获取实体名称
            start_name = self.extractor.get_cui_name(start_cui) if start_cui else None
            end_name = self.extractor.get_cui_name(end_cui) if end_cui else None

            cui_pair = {
                "start": {
                    "cui": start_cui,
                    "question": pair['start'],
                    "matched_name": start_name,
                    "confidence": round(start_score, 3)
                },
                "end": {
                    "cui": end_cui,
                    "question": pair['end'],
                    "matched_name": end_name,
                    "confidence": round(end_score, 3)
                },
                "reasoning": pair['reasoning']
            }
            processed_pairs.append(cui_pair)

        if 'analysis_result' in processed_data:
            processed_data['analysis_result']['entity_pairs'] = processed_pairs

        return processed_data

    def process_file(self, input_file: str, output_file: str):
        """
        处理输入文件并保存结果

        Args:
            input_file: 输入JSON文件路径
            output_file: 输出JSON文件路径
        """
        try:
            # 读取输入数据
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                data = [data]

            # 处理每个问题
            processed_questions = []
            unmatched_entities = []

            for question in data:
                processed_question = self.convert_entity_pairs_to_cuis(question)
                processed_questions.append(processed_question)

                # 收集未匹配的实体
                for pair in processed_question['analysis_result']['entity_pairs']:
                    if not pair['start']['cui']:
                        unmatched_entities.append(pair['start']['question'])
                    if not pair['end']['cui']:
                        unmatched_entities.append(pair['end']['question'])

            # 保存结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_questions, f, indent=2, ensure_ascii=False)

            # 打印统计信息
            if unmatched_entities:
                print("\n未能找到CUI的实体:")
                for entity in set(unmatched_entities):
                    print(f"- {entity}")

            return processed_questions

        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            return None


def main():
    # 示例用法
    converter = EntityCUIConverter(threshold=0.7)
    input_file = "../temp/pairs.json"
    output_file = "../temp/pairs_with_entity.json"

    processed_data = converter.process_file(input_file, output_file)

    if processed_data:
        print("\n转换示例:")
        example_pairs = processed_data[0]['analysis_result']['entity_pairs']
        for pair in example_pairs:
            print(f"\n实体对转换结果:")
            print(f"起始实体:")
            print(f"  原文: {pair['start']['question']}")
            print(f"  CUI: {pair['start']['cui']}")
            print(f"  匹配名称: {pair['start']['matched_name']}")
            print(f"  置信度: {pair['start']['confidence']}")

            print(f"终止实体:")
            print(f"  原文: {pair['end']['question']}")
            print(f"  CUI: {pair['end']['cui']}")
            print(f"  匹配名称: {pair['end']['matched_name']}")
            print(f"  置信度: {pair['end']['confidence']}")


if __name__ == "__main__":
    main()