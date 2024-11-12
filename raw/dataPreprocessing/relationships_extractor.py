def extract_relation_names(file_path):
    """
    从RRF文件中提取所有唯一的关系名称

    Args:
        file_path (str): RRF文件的路径

    Returns:
        set: 包含所有唯一关系名称的集合
    """
    relation_names = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 将行按竖线分割
                fields = line.strip().split('|')

                # RRF格式中关系名称在第8个字段（索引7）
                if len(fields) > 7 and fields[7]:  # 确保字段存在且非空
                    relation_name = fields[7].strip()
                    if relation_name:  # 只添加非空的关系名称
                        relation_names.add(relation_name)

        print(f"\n找到的唯一关系名称（共 {len(relation_names)} 个）:")
        for name in sorted(relation_names):
            print(f"- {name}")

        return relation_names

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")

    return set()


# 使用示例
if __name__ == "__main__":
    file_path = '../data/database/raw/META/MRREL.RRF'  # 替换为你的实际文件路径
    relations = extract_relation_names(file_path)

    # 可以选择将结果保存到文件
    if relations:
        output_path = '../data/database/Processed/umls/relation_names.txt'
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for relation in sorted(relations):
                    f.write(relation + '\n')
            print(f"\n关系名称已保存到: {output_path}")
        except Exception as e:
            print(f"保存文件时出错: {str(e)}")