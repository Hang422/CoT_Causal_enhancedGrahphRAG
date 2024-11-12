import csv
import sys
from collections import defaultdict
import os
from tqdm import tqdm

# 增加CSV字段大小限制
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)


class UMLSCausalFilter:
    def __init__(self):
        # 定义因果关系类型
        self.causal_relations = {
            # 核心因果关系
            'cause_of', 'due_to',
            'has_causative_agent', 'causative_agent_of',
            'induces', 'induced_by',

            # 调控关系
            'regulates', 'regulated_by',
            'positively_regulates', 'positively_regulated_by',
            'negatively_regulates', 'negatively_regulated_by',

            # 疾病标志物关系
            'disease_has_finding',
            'disease_is_marked_by_gene'
        }
        self.related_cuis = set()  # 存储涉及因果关系的CUI

    def filter_relations(self, mrrel_path, output_dir):
        """提取因果关系"""
        os.makedirs(output_dir, exist_ok=True)
        output_rel_path = os.path.join(output_dir, 'causal_MRREL.RRF')
        relation_stats = defaultdict(int)

        print("Extracting causal relations from MRREL.RRF...")
        try:
            with open(mrrel_path, 'r', encoding='utf-8') as infile, \
                    open(output_rel_path, 'w', encoding='utf-8', newline='') as outfile:

                reader = csv.reader(infile, delimiter='|')
                writer = csv.writer(outfile, delimiter='|')

                for row in tqdm(reader, desc="Filtering relations"):
                    try:
                        if len(row) > 7:  # 确保行有足够的字段
                            rela = row[7]  # RELA字段
                            if rela:
                                rela = rela.upper().replace(' ', '_')

                                # 检查是否是因果关系
                                if rela in self.causal_relations:
                                    writer.writerow(row)
                                    relation_stats[rela] += 1

                                    # 记录涉及的CUI
                                    self.related_cuis.add(row[0])  # 源概念
                                    self.related_cuis.add(row[4])  # 目标概念
                    except Exception as e:
                        print(f"Error processing row: {e}")
                        continue

            # 保存统计信息
            stats_file = os.path.join(output_dir, 'relation_statistics.txt')
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write("Causal Relation Statistics:\n")
                f.write("-" * 30 + "\n")
                total = sum(relation_stats.values())
                f.write(f"Total causal relations: {total}\n")
                f.write(f"Unique concepts involved: {len(self.related_cuis)}\n\n")
                f.write("Breakdown by relation type:\n")
                for rel, count in sorted(relation_stats.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{rel}: {count} ({(count / total * 100):.2f}%)\n")

            print("\nFiltering completed!")
            print(f"Found {total} causal relations involving {len(self.related_cuis)} unique concepts")
            print(f"\nOutput files:")
            print(f"1. Filtered relations: {output_rel_path}")
            print(f"2. Statistics: {stats_file}")

            return output_rel_path

        except Exception as e:
            print(f"Error filtering relations: {e}")

    def filter_concepts(self, mrconso_path, output_dir):
        """只提取涉及因果关系的概念"""
        output_con_path = os.path.join(output_dir, 'causal_MRCONSO.RRF')
        concept_stats = defaultdict(int)

        print("\nExtracting relevant concepts from MRCONSO.RRF...")
        try:
            with open(mrconso_path, 'r', encoding='utf-8') as infile, \
                    open(output_con_path, 'w', encoding='utf-8', newline='') as outfile:

                reader = csv.reader(infile, delimiter='|')
                writer = csv.writer(outfile, delimiter='|')

                for row in tqdm(reader, desc="Filtering concepts"):
                    try:
                        if len(row) > 14:  # 确保行有足够的字段
                            cui = row[0]
                            lang = row[1]

                            # 只保存涉及因果关系的概念的英文名称
                            if cui in self.related_cuis and lang == 'ENG':
                                writer.writerow(row)
                                concept_stats[cui] += 1
                    except Exception as e:
                        print(f"Error processing concept row: {e}")
                        continue

            # 更新统计信息
            stats_file = os.path.join(output_dir, 'relation_statistics.txt')
            with open(stats_file, 'a', encoding='utf-8') as f:
                f.write("\nConcept Statistics:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total concepts saved: {len(concept_stats)}\n")
                f.write(f"Total concept terms: {sum(concept_stats.values())}\n")
                f.write(f"Average terms per concept: {sum(concept_stats.values()) / len(concept_stats):.2f}\n")

            print(f"Extracted {sum(concept_stats.values())} terms for {len(concept_stats)} concepts")
            print(f"Filtered concepts saved to: {output_con_path}")

            return output_con_path

        except Exception as e:
            print(f"Error filtering concepts: {e}")


def main():
    # 初始化过滤器
    filter = UMLSCausalFilter()

    # 设置输入输出路径
    umls_dir = "../data/database/raw/META/"
    output_dir = "../data/database/Processed/"

    # 首先过滤关系文件
    rel_file = filter.filter_relations(
        os.path.join(umls_dir, "MRREL.RRF"),
        output_dir
    )

    # 然后过滤概念文件
    #if rel_file:  # 只有在关系过滤成功时才继续
    #    con_file = filter.filter_concepts(
    #        os.path.join(umls_dir, "MRCONSO.RRF"),
    #        output_dir
    #    )


if __name__ == "__main__":
    main()
