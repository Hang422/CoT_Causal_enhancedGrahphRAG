def preprocess_rrf(input_file, output_file):
    """
    预处理RRF文件，提取关键信息并正确处理空字符

    Args:
        input_file (str): 输入RRF文件路径
        output_file (str): 输出文件路径
    """
    try:
        processed_data = []
        skipped_lines = 0
        empty_relation_lines = 0

        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 分割每行数据
                fields = line.strip().split('|')

                # 确保至少有必要的字段
                if len(fields) < 9:  # RRF至少应该有8个字段加关系ID
                    skipped_lines += 1
                    continue

                # 提取并清理字段
                entity1_cui = fields[0].strip()
                entity1_name = fields[1].strip()
                entity2_cui = fields[4].strip()
                entity2_name = fields[5].strip()
                relation_name = fields[7].strip()
                relation_id = fields[8].strip()

                # 验证必要字段
                if not all([entity1_cui, entity2_cui, relation_id]):  # CUI和关系ID是必需的
                    skipped_lines += 1
                    continue

                # 处理空的关系名
                if not relation_name:
                    empty_relation_lines += 1
                    continue

                # 规范化空字段
                entity1_name = entity1_name if entity1_name else "UNKNOWN"
                entity2_name = entity2_name if entity2_name else "UNKNOWN"

                # 构建输出行
                processed_line = {
                    'entity1_cui': entity1_cui,
                    'entity1_name': entity1_name,
                    'entity2_cui': entity2_cui,
                    'entity2_name': entity2_name,
                    'relation_name': relation_name,
                    'relation_id': relation_id
                }

                processed_data.append(processed_line)

        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入表头
            header = "Entity1_CUI\tEntity1_Name\tEntity2_CUI\tEntity2_Name\tRelation_Name\tRelation_ID\n"
            f.write(header)

            # 写入数据
            for item in processed_data:
                line = f"{item['entity1_cui']}\t{item['entity1_name']}\t" \
                       f"{item['entity2_cui']}\t{item['entity2_name']}\t" \
                       f"{item['relation_name']}\t{item['relation_id']}\n"
                f.write(line)

        # 打印处理统计
        print("\n=== 处理统计 ===")
        print(f"总行数: {line_num}")
        print(f"成功处理: {len(processed_data)} 行")
        print(f"跳过的行数: {skipped_lines}")
        print(f"空关系名的行数: {empty_relation_lines}")
        print(f"结果已保存到: {output_file}")

        # 显示数据示例
        print("\n=== 数据示例（前3行）===")
        for i, item in enumerate(processed_data[:3], 1):
            print(f"\n示例 {i}:")
            print(f"Entity1_CUI: {item['entity1_cui']}")
            print(f"Entity1_Name: {item['entity1_name']}")
            print(f"Entity2_CUI: {item['entity2_cui']}")
            print(f"Entity2_Name: {item['entity2_name']}")
            print(f"Relation_Name: {item['relation_name']}")
            print(f"Relation_ID: {item['relation_id']}")

        return processed_data

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        print(traceback.format_exc())


def analyze_relationships(processed_data):
    """
    分析处理后的关系数据

    Args:
        processed_data (list): 处理后的关系数据列表
    """
    if not processed_data:
        print("没有数据可供分析")
        return

    # 统计信息
    relation_counts = {}
    entity_pairs = set()
    unique_entities = set()

    for item in processed_data:
        # 统计关系
        relation = item['relation_name']
        relation_counts[relation] = relation_counts.get(relation, 0) + 1

        # 统计实体对
        entity_pair = (item['entity1_cui'], item['entity2_cui'])
        entity_pairs.add(entity_pair)

        # 统计唯一实体
        unique_entities.add(item['entity1_cui'])
        unique_entities.add(item['entity2_cui'])

    print("\n=== 关系分析 ===")
    print(f"唯一关系类型数: {len(relation_counts)}")
    print(f"唯一实体对数: {len(entity_pairs)}")
    print(f"唯一实体数: {len(unique_entities)}")

    print("\n最常见的10种关系:")
    for relation, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{relation}: {count}次")


# 使用示例
if __name__ == "__main__":
    input_file = "../data/database/raw/META/MRREL.RRF"  # 替换为你的输入文件路径
    output_file = "../data/database/Processed/umls/total_relations.csv"  # 输出文件

    # 处理数据
    processed_data = preprocess_rrf(input_file, output_file)

