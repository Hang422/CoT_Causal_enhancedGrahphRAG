import pandas as pd
import spacy
import scispacy
from scispacy.linking import EntityLinker
import en_core_sci_md
import json
from tqdm import tqdm  # 添加进度条


def setup_nlp():
    """设置NLP模型和实体链接器"""
    nlp = en_core_sci_md.load()
    if 'scispacy_linker' not in nlp.pipe_names:
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    return nlp


def get_filtered_cuis(text, nlp, threshold=1.00):
    """从文本中提取CUI列表，仅保留高于阈值的匹配"""
    doc = nlp(text)
    cuis = []
    for ent in doc.ents:
        if ent._.kb_ents:
            # kb_ents是一个包含(CUI, 分数)元组的列表
            for cui, score in ent._.kb_ents:
                if score >= threshold:
                    cuis.append(cui)
                    break  # 只取第一个超过阈值的CUI
    return list(set(cuis))  # 去重


def process_csv_to_json(input_csv, output_json, threshold=0.75):
    """处理CSV文件并将过滤后的CUI列表输出到JSON"""
    print(f"Processing CSV file with confidence threshold: {threshold}")

    # 读取CSV
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from CSV")

    # 设置NLP
    nlp = setup_nlp()

    # 存储结果
    results = []

    # 使用tqdm添加进度条
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        try:
            # 获取问题的CUIs
            question_cuis = get_filtered_cuis(row['question'], nlp, threshold)

            # 合并所有选项的文本并获取CUIs
            options_text = f"{row['opa']} {row['opb']} {row['opc']} {row['opd']}"
            options_cuis = get_filtered_cuis(options_text, nlp, threshold)

            # 创建当前行的结果
            result = {
                "question_cuis": question_cuis,
                "options_cuis": options_cuis
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            results.append({
                "question_cuis": [],
                "options_cuis": [],
                "error": str(e)
            })

    # 统计信息
    total_questions = len([r for r in results if r['question_cuis']])
    total_options = len([r for r in results if r['options_cuis']])
    print(f"\nStatistics:")
    print(f"Total rows processed: {len(results)}")
    print(f"Questions with CUIs: {total_questions}")
    print(f"Options with CUIs: {total_options}")

    # 写入JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_json}")
    return results


if __name__ == "__main__":
    # 使用示例
    input_csv = '../data/testdata/samples.csv'
    output_json = '../data/testdata/dense_sample_cui.json'
    threshold = 0.75  # 设置匹配分数阈值

    results = process_csv_to_json(input_csv, output_json, threshold)

    # 打印第一个非空结果作为示例
    print("\nSample output format:")
    for result in results:
        if result['question_cuis'] or result['options_cuis']:
            print(json.dumps(result, indent=2))
            break