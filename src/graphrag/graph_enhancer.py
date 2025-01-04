from typing import List, Dict, Set, Tuple
import logging
from dataclasses import dataclass
import math

from src.modules.MedicalQuestion import MedicalQuestion
from src.graphrag.entity_processor import EntityProcessor


@dataclass
class PathScore:
    """Path scoring information"""
    path: str
    cui_match_score: float
    semantic_match_score: float
    length_score: float  # Normalized path length score (shorter is better)
    total_score: float
    path_length: int  # Original path length for reference


class EnhancedGraphEnhancer:
    def __init__(self, keep_ratio: float = 0.6):
        self.logger = logging.getLogger(__name__)
        self.entity_processor = EntityProcessor()
        self.keep_ratio = keep_ratio

    def enhance_graphs(self, question: MedicalQuestion) -> None:
        try:
            # Merge paths from both graphs
            all_paths = []
            all_paths.extend(question.causal_graph.paths)
            all_paths.extend(question.knowledge_graph.paths)

            # Group similar paths
            path_groups = {}
            for path in all_paths:
                entities, relations, intermediates = self._extract_path_elements(path)

                # Create key based on structure
                key = (
                    entities[0],  # start entity
                    entities[-1],  # end entity
                    tuple(sorted(set(intermediates)))  # unique sorted intermediate nodes
                )

                if key not in path_groups:
                    path_groups[key] = {
                        'paths': [],
                        'relations': [set() for _ in range(len(relations))]
                    }

                # Add path to group
                path_groups[key]['paths'].append(path)

                # Add relations at each position
                for i, rel in enumerate(relations):
                    path_groups[key]['relations'][i].add(rel)

            # Merge paths in each group
            merged_paths = []
            for key, group_info in path_groups.items():

                if len(group_info['paths']) > 1:
                    # Process group with multiple paths
                    merged = self._merge_path_group(group_info['paths'], group_info['relations'])
                    merged_paths.append(merged)
                else:
                    # Keep single path as is
                    merged_paths.append(group_info['paths'][0])


            # Score and select paths
            selected_paths = self._score_and_select_paths(merged_paths, question)
            question.enhanced_graph.paths = selected_paths


        except Exception as e:
            self.logger.error(f"Error in graph enhancement: {str(e)}", exc_info=True)
            question.enhanced_graph.paths = []

    def _extract_path_elements(self, path: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract entities, relations, and intermediates from a path"""
        parts = path.split('->')
        entities = []
        relations = []
        intermediates = []

        for i, part in enumerate(parts):
            # Extract entity
            if '(' in part and ')' in part:
                entity_part = part[part.find('(') + 1:part.find(')')]
                # Handle multiple entities
                curr_entities = [e.strip() for e in entity_part.split(' and ')]
                entities.extend(curr_entities)

                # Add to intermediates if not start/end
                if 0 < i < len(parts) - 1:
                    intermediates.extend(curr_entities)

            # Extract relation
            if '-' in part:
                relation = part.split('-')[1]
                relations.append(relation)

        return entities, relations, intermediates

    def _merge_path_group(self, paths: List[str], relations_by_pos: List[Set[str]]) -> str:
        """Merge a group of paths with similar structure"""
        parts_by_position = []
        example_path = paths[0]
        path_parts = example_path.split('->')

        # Initialize collection for each position
        for _ in range(len(path_parts)):
            parts_by_position.append({
                'entities': set(),
                'relations': set()
            })

        # Collect all entities and relations
        for path in paths:
            parts = path.split('->')
            for i, part in enumerate(parts):
                # Extract entities
                if '(' in part and ')' in part:
                    entity_part = part[part.find('(') + 1:part.find(')')]
                    for entity in entity_part.split(' and '):
                        parts_by_position[i]['entities'].add(entity.strip())

                # Extract relations
                if '-' in part:
                    relation = part.split('-')[1]
                    parts_by_position[i]['relations'].add(relation)

        # Build merged path
        merged_parts = []
        for i, part_info in enumerate(parts_by_position):
            if part_info['entities']:
                entities_str = ' and '.join(sorted(part_info['entities']))
                if part_info['relations']:
                    relations_str = '/'.join(sorted(part_info['relations']))
                    merged_parts.append(f"({entities_str})-{relations_str}")
                else:
                    merged_parts.append(f"({entities_str})")

        return '->'.join(merged_parts)

    def _score_and_select_paths(self, paths: List[str], question: MedicalQuestion) -> List[str]:
        """Score paths and select top ones based on relevance"""
        # Extract question entities
        question_cuis = self.entity_processor.extract_cuis_from_text(question.question)
        for option_text in question.options.values():
            question_cuis.update(self.entity_processor.extract_cuis_from_text(option_text))

        question_semantic_types = set()
        for cui in question_cuis:
            question_semantic_types.update(
                self.entity_processor.get_semantic_types_for_cui(cui))

        # Score paths
        path_scores = []
        for path in paths:
            path_cuis, path_stypes = self._extract_path_entities(path)
            path_length = len(path.split('->'))

            # Calculate scores
            cui_score = self._calculate_overlap_score(path_cuis, question_cuis)
            semantic_score = self._calculate_overlap_score(path_stypes, question_semantic_types)
            length_score = 1.0 / path_length  # Shorter paths get higher scores

            total_score = (cui_score * 0.4 + semantic_score * 0.4 + length_score * 0.2)
            path_scores.append((path, total_score))

        # Sort and select top paths
        path_scores.sort(key=lambda x: x[1], reverse=True)
        keep_count = max(1, int(len(paths) * self.keep_ratio))
        return [score[0] for score in path_scores[:keep_count]]

    def _extract_path_entities(self, path: str) -> Tuple[Set[str], Set[str]]:
        """Extract CUIs and semantic types from path"""
        entities = []
        parts = path.split('->')
        for part in parts:
            if '(' in part and ')' in part:
                entity = part[part.find('(') + 1:part.find(')')].strip()
                for sub_entity in entity.split(' and '):
                    entities.append(sub_entity.strip())

        path_cuis = set()
        for entity in entities:
            path_cuis.update(self.entity_processor.extract_cuis_from_text(entity))

        path_stypes = set()
        for cui in path_cuis:
            path_stypes.update(self.entity_processor.get_semantic_types_for_cui(cui))

        return path_cuis, path_stypes

    def _calculate_overlap_score(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate overlap score between two sets"""
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set2)


def extract_path_elements(path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract entities, relations, and intermediate entities from a path
    Returns:
        Tuple of (entities, relations, intermediate_nodes)
    """
    parts = path.split('->')
    entities = []
    relations = []
    intermediate_nodes = []

    for i, part in enumerate(parts):
        # Extract main entity for this part
        if '(' in part and ')' in part:
            # Handle potential "and" separated entities
            entity_part = part[part.find('(') + 1:part.find(')')]
            main_entities = [e.strip() for e in entity_part.split(' and ')]

            # Add all entities at this position
            for entity in main_entities:
                entities.append(entity)
                if 0 < i < len(parts) - 1:  # If it's an intermediate node
                    intermediate_nodes.append(entity)

        # Extract relation if present
        if '-' in part:
            relation = part.split('-')[1]
            relations.append(relation)

    return entities, relations, intermediate_nodes


def merge_group(paths: List[str]) -> str:
    """
    Merge a group of paths with the same structure
    """
    if len(paths) == 1:
        return paths[0]

    parts_by_position = []
    example_path = paths[0]
    path_parts = example_path.split('->')

    # Initialize collection for each position
    for _ in range(len(path_parts)):
        parts_by_position.append({
            'entities': set(),
            'relation': None
        })

    # Collect all entities and relations at each position
    for path in paths:
        parts = path.split('->')
        for i, part in enumerate(parts):
            current = parts_by_position[i]

            # Extract entity
            if '(' in part and ')' in part:
                entity_part = part[part.find('(') + 1:part.find(')')]
                # Handle multiple entities separated by 'and'
                for entity in entity_part.split(' and '):
                    current['entities'].add(entity.strip())

            # Extract relation
            if '-' in part:
                relation = part.split('-')[1]
                if current['relation'] is None:
                    current['relation'] = relation
                elif current['relation'] != relation:
                    # If we find different relations, don't merge
                    return paths[0]

    # Build merged path
    merged_parts = []
    for i, part_info in enumerate(parts_by_position):
        if part_info['entities']:
            entities_str = ' and '.join(sorted(part_info['entities']))
            if part_info['relation']:
                merged_parts.append(f"({entities_str})-{part_info['relation']}")
            else:
                merged_parts.append(f"({entities_str})")

    return '->'.join(merged_parts)




def main():
    """测试增强器功能"""
    # 创建测试用例
    question = MedicalQuestion(
        question= "All of the following are true about Sickle cell disease, Except:",
    is_multi_choice= True,
    correct_answer= "opc",
    options= {
        "opa": "Single nucleotide change results in change of Glutamine to Valine",
        "opb": "RFLP results from a single base change",
        "opc": "'Sticky patch' is generated as a result of replacement of a non polar residue with a polar residue",
        "opd": "HbS confers resistance against malaria in heterozygotes"
    }
    )

    # 添加测试路径
    question.causal_graph.paths = [
      "(Mandibular right second primary molar)-LOCATION_OF->(Infection)-PREDISPOSES->(Pathogenesis)-ASSOCIATED_WITH->(Valine)",
      "(Mandibular right second primary molar)-LOCATION_OF->(Infection)-MANIFESTATION_OF->(Pathogenesis)-ASSOCIATED_WITH->(Valine)",
      "(Mandibular right second primary molar)-LOCATION_OF->(Infection)-CAUSES->(Pathogenesis)-ASSOCIATED_WITH->(Valine)",
      "(Mandibular right second primary molar)-LOCATION_OF->(Infection)-PREDISPOSES->(Pathogenesis)-ASSOCIATED_WITH->(Glutamic Acid)",
      "(Mandibular right second primary molar)-LOCATION_OF->(Infection)-MANIFESTATION_OF->(Pathogenesis)-ASSOCIATED_WITH->(Glutamic Acid)",
      "(Mandibular right second primary molar)-LOCATION_OF->(Infection)-CAUSES->(Pathogenesis)-ASSOCIATED_WITH->(Glutamic Acid)",
      "(Valine)-INTERACTS_WITH->(Cells)-LOCATION_OF->(Glutamine)",
      "(Valine)-TREATS->(Functional disorder)-ASSOCIATED_WITH->(Glutamine)",
      "(Valine)-INTERACTS_WITH->(Cells)-INTERACTS_WITH->(Glutamine)",
      "(Glutamic Acid)-INTERACTS_WITH->(Cells and Chinese Hamster Ovary Cell)-INTERACTS_WITH->(Valine)",
      "(Glutamic Acid)-CAUSES->(Excretory function)-ASSOCIATED_WITH->(Valine)",
      "(Glutamic Acid)-INTERACTS_WITH->(Cells)-LOCATION_OF->(Glutamine)",
      "(Glutamic Acid)-INTERACTS_WITH->(Cells and PC12 Cells)-INTERACTS_WITH->(Glutamine)",
      "(Sickle Cell Anemia)-PREDISPOSES->(Pathogenesis)-ASSOCIATED_WITH->(Valine)",
      "(Sickle Cell Anemia)-MANIFESTATION_OF->(Pathogenesis)-ASSOCIATED_WITH->(Valine)",
      "(Sickle Cell Anemia)-ISA->(Pathogenesis)-ASSOCIATED_WITH->(Valine)",
      "(Sickle Cell Anemia)-PREDISPOSES->(Pathogenesis)-ASSOCIATED_WITH->(Glutamic Acid)",
      "(Sickle Cell Anemia)-MANIFESTATION_OF->(Pathogenesis)-ASSOCIATED_WITH->(Glutamic Acid)",
      "(Sickle Cell Anemia)-ISA->(Pathogenesis)-ASSOCIATED_WITH->(Glutamic Acid)",
      "(Sickle Hemoglobin)-INTERACTS_WITH->(Cells and Hematopoietic stem cells)-PART_OF->(Arterial Media and Fetal Tissue)-LOCATION_OF->(Surgical Replantation)",
      "(Polymerization)-CAUSES->(Adhesions and Thrombus)-ASSOCIATED_WITH->(Genes)-PART_OF->(Cartilage and Ligaments)-LOCATION_OF->(Surgical Replantation)",
      "(Sickle Hemoglobin)-CAUSES->(Sickle Cell Anemia)",
      "(Sickle Cell Anemia)-CAUSES->(Hypoxia)-ASSOCIATED_WITH->(Sickle Hemoglobin)",
      "(Sickle Cell Anemia)-ISA->(Complication)-ASSOCIATED_WITH->(Sickle Hemoglobin)",
      "(Sickle Cell Anemia)-PREDISPOSES->(Hypoxia)-ASSOCIATED_WITH->(Sickle Hemoglobin)",
      "(Sickle Cell Anemia)-CAUSES->(Sickle Cell Trait)",
      "(Abnormal Hemoglobins)-INTERACTS_WITH->(Erythrocytes)-INTERACTS_WITH->(Sickle Hemoglobin)",
      "(Abnormal Hemoglobins)-CAUSES->(Symptoms)-CAUSES->(Malaria)-CAUSES->(Sickle Cell Trait)",
      "(Sickle Hemoglobin)-CAUSES->(Symptoms)-CAUSES->(Malaria)",
      "(Sickle Cell Trait)-PREDISPOSES->(Malaria)",
      "(Malaria)-CAUSES->(Complication and Hypoxia)-ASSOCIATED_WITH->(Sickle Hemoglobin)",
      "(Malaria)-PREDISPOSES->(Complication)-ASSOCIATED_WITH->(Sickle Hemoglobin)"
    ]

    # 创建并运行增强器
    enhancer = EnhancedGraphEnhancer(keep_ratio=0.6)
    print("\nBefore enhancement:")
    print(f"Number of causal graph paths: {len(question.causal_graph.paths)}")
    print(f"Number of knowledge graph paths: {len(question.knowledge_graph.paths)}")

    enhancer.enhance_graphs(question)

    print("\nAfter enhancement:")
    print(f"Number of enhanced paths: {len(question.enhanced_graph.paths)}")
    print("\nEnhanced paths:")
    for path in question.enhanced_graph.paths:
        print(path)


if __name__ == "__main__":
    main()
