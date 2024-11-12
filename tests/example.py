import pandas as pd
from src.modules.data_format import Question, DataFormat, OptionKey, FinalReasoning


f = pd.read_csv('../data/testdata/test_sample.csv')
question = Question(
    text=f['question'][0],
    options={
        OptionKey.A: f.iloc[0]['opa'],
        OptionKey.B: f.iloc[0]['opb'],
        OptionKey.C: f.iloc[0]['opc'],
        OptionKey.D: f.iloc[0]['opd'],
    },
    correct_answer=(
        OptionKey.A if f['cop'][0] == 0 else
        OptionKey.B if f['cop'][0] == 1 else
        OptionKey.C if f['cop'][0] == 2 else
        OptionKey.D if f['cop'][0] == 3 else
        None
    )
)

# 创建 DataFormat.QuestionGraphResult 实例
question_graph_result = DataFormat.QuestionGraphResult(
    question_paths=[],
    option_paths={
        OptionKey.A: [],
        OptionKey.B: [
            DataFormat.GraphPath(
                nodes=['C0027750', 'C0001811', 'C0002520', 'C0746922'],
                node_names=['Myelinated nerve fiber', 'Aging', 'Amino Acids', 'Anatomic Node'],
                relationships=['AFFECTS', 'PRODUCES', 'DISRUPTS'],
                source_entity=DataFormat.EntityInfo(name='Myelinated nerve fiber', types=['T026'], cui='C0027750'),
                target_entity=DataFormat.EntityInfo(name='Anatomic Node', types=['T023'], cui='C0746922')
            ),
            DataFormat.GraphPath(
                nodes=['C0027750', 'C0001811', 'C0006826', 'C0746922'],
                node_names=['Myelinated nerve fiber', 'Aging', 'Malignant Neoplasms', 'Anatomic Node'],
                relationships=['AFFECTS', 'AFFECTS', 'AFFECTS'],
                source_entity=DataFormat.EntityInfo(name='Myelinated nerve fiber', types=['T026'], cui='C0027750'),
                target_entity=DataFormat.EntityInfo(name='Anatomic Node', types=['T023'], cui='C0746922')
            ),
            # 继续添加其他 GraphPath 实例...
        ],
        OptionKey.C: [
            DataFormat.GraphPath(
                nodes=['C0027750', 'C0001811', 'C0003811', 'C0232217'],
                node_names=['Myelinated nerve fiber', 'Aging', 'Cardiac Arrhythmia', 'Cardiac conduction'],
                relationships=['AFFECTS', 'AFFECTS', 'AFFECTS'],
                source_entity=DataFormat.EntityInfo(name='Myelinated nerve fiber', types=['T026'], cui='C0027750'),
                target_entity=DataFormat.EntityInfo(name='Cardiac conduction', types=['T042'], cui='C0232217')
            ),
            # 继续添加其他 GraphPath 实例...
        ],
        OptionKey.D: [
            DataFormat.GraphPath(
                nodes=['C0027750', 'C0014063', 'C0027740'],
                node_names=['Myelinated nerve fiber', 'Myelin Basic Proteins', 'Nerve'],
                relationships=['PRODUCES', 'AFFECTS'],
                source_entity=DataFormat.EntityInfo(name='Myelinated nerve fiber', types=['T026'], cui='C0027750'),
                target_entity=DataFormat.EntityInfo(name='Nerve', types=['T023'], cui='C0027740')
            ),
            # 继续添加其他 GraphPath 实例...
        ]
    }
)


from src.modules.data_format import DataFormat

# 创建 DataFormat.QuestionAnalysis 实例
question_analysis = DataFormat.QuestionAnalysis(
    reasoning_chain=[
        "step1: Identify the characteristics of myelinated nerve fibers and their conduction properties compared to non-myelinated fibers.",
        "step2: Understand the role of the nodes of Ranvier in myelinated fibers and how they contribute to impulse conduction.",
        "step3: Analyze the concept of saltatory conduction and how it differs from continuous conduction in non-myelinated fibers.",
        "step4: Investigate the mechanism of local anesthesia and its effectiveness on myelinated versus non-myelinated fibers."
    ],
    entity_pairs=[
        DataFormat.EntityPairs(
            start=["myelinated nerve fibers", "myelinated axons"],
            end=["non-myelinated nerve fibers", "unmyelinated axons"],
            reasoning="Understanding the differences in impulse conduction speed between myelinated and non-myelinated fibers is crucial to determine the accuracy of option a."
        ),
        DataFormat.EntityPairs(
            start=["nodes of Ranvier", "Ranvier nodes"],
            end=["membrane currents", "ionic currents"],
            reasoning="Investigating the generation of membrane currents at the nodes of Ranvier will clarify the validity of option b."
        ),
        DataFormat.EntityPairs(
            start=["saltatory conduction", "saltatory transmission"],
            end=["impulse conduction", "nerve impulse propagation"],
            reasoning="Examining the process of saltatory conduction will help verify the correctness of option c."
        ),
        DataFormat.EntityPairs(
            start=["local anesthesia", "anesthetic agents"],
            end=["myelin sheath", "myelin"],
            reasoning="Understanding how local anesthesia interacts with myelinated fibers will provide insight into the accuracy of option d."
        )
    ]
)

from src.modules.data_format import DataFormat

# 创建 entity_pairs 列表
entity_pairs = [
    (
        DataFormat.EntityPairs(
            start=['myelinated nerve fibers', 'myelinated axons'],
            end=['non-myelinated nerve fibers', 'unmyelinated axons'],
            reasoning='Understanding the differences in impulse conduction speed between myelinated and non-myelinated fibers is crucial to determine the accuracy of option a.'
        ),
        [
            DataFormat.GraphPath(
                nodes=['C0027750', 'C0501409'],
                node_names=['Myelinated nerve fiber', 'Nerve Fibers, Unmyelinated'],
                relationships=['PART_OF'],
                source_entity=DataFormat.EntityInfo(name='Myelinated nerve fiber', types=['T026'], cui='C0027750'),
                target_entity=DataFormat.EntityInfo(name='Nerve Fibers, Unmyelinated', types=['T026'], cui='C0501409')
            ),
            DataFormat.GraphPath(
                nodes=['C0026973', 'C0034693', 'C0501409'],
                node_names=['Myelin Sheath', 'Rattus norvegicus', 'Nerve Fibers, Unmyelinated'],
                relationships=['PART_OF', 'PART_OF'],
                source_entity=DataFormat.EntityInfo(name='Myelin Sheath', types=['T026'], cui='C0026973'),
                target_entity=DataFormat.EntityInfo(name='Nerve Fibers, Unmyelinated', types=['T026'], cui='C0501409')
            ),
            DataFormat.GraphPath(
                nodes=['C0004461', 'C0501409'],
                node_names=['Axon', 'Nerve Fibers, Unmyelinated'],
                relationships=['PART_OF'],
                source_entity=DataFormat.EntityInfo(name='Axon', types=['T026'], cui='C0004461'),
                target_entity=DataFormat.EntityInfo(name='Nerve Fibers, Unmyelinated', types=['T026'], cui='C0501409')
            )
        ]
    ),
    (
        DataFormat.EntityPairs(
            start=['nodes of Ranvier', 'Ranvier nodes'],
            end=['membrane currents', 'ionic currents'],
            reasoning='Investigating the generation of membrane currents at the nodes of Ranvier will clarify the validity of option b.'
        ),
        []  # 空的验证路径列表
    ),
    (
        DataFormat.EntityPairs(
            start=['saltatory conduction', 'saltatory transmission'],
            end=['impulse conduction', 'nerve impulse propagation'],
            reasoning='Examining the process of saltatory conduction will help verify the correctness of option c.'
        ),
        [
            DataFormat.GraphPath(
                nodes=['C0234093', 'C0034693', 'C0027740'],
                node_names=['Saltatory conduction', 'Rattus norvegicus', 'Nerve'],
                relationships=['PROCESS_OF', 'PART_OF'],
                source_entity=DataFormat.EntityInfo(name='Saltatory conduction', types=['T043'], cui='C0234093'),
                target_entity=DataFormat.EntityInfo(name='Nerve', types=['T023'], cui='C0027740')
            )
        ]
    ),
    (
        DataFormat.EntityPairs(
            start=['local anesthesia', 'anesthetic agents'],
            end=['myelin sheath', 'myelin'],
            reasoning='Understanding how local anesthesia interacts with myelinated fibers will provide insight into the accuracy of option d.'
        ),
        [
            DataFormat.GraphPath(
                nodes=['C0002921', 'C0034693', 'C0026973'],
                node_names=['Local anesthesia', 'Rattus norvegicus', 'Myelin Sheath'],
                relationships=['TREATS', 'PART_OF'],
                source_entity=DataFormat.EntityInfo(name='Local anesthesia', types=['T061'], cui='C0002921'),
                target_entity=DataFormat.EntityInfo(name='Myelin Sheath', types=['T026'], cui='C0026973')
            ),
            DataFormat.GraphPath(
                nodes=['C0002932', 'C0013081', 'C0026973'],
                node_names=['Anesthetics', 'Down-Regulation', 'Myelin Sheath'],
                relationships=['AUGMENTS', 'LOCATION_OF'],
                source_entity=DataFormat.EntityInfo(name='Anesthetics', types=['T121'], cui='C0002932'),
                target_entity=DataFormat.EntityInfo(name='Myelin Sheath', types=['T026'], cui='C0026973')
            )
        ]
    )
]


final_reasoning = FinalReasoning(
    question=question,
    initial_paths=None,
    entity_pairs=entity_pairs,
    reasoning_chain=question_analysis.reasoning_chain
)



