import spacy
import scispacy
from scispacy.linking import EntityLinker
import en_core_sci_md
import warnings
from typing import List, Dict, Optional, Set, Tuple


class EntityProcessor:
    """Medical entity processor for converting between entity names and UMLS CUIs"""

    def __init__(self, threshold: float = 0.9):
        """
        Initialize the entity processor

        Args:
            threshold: Confidence threshold for entity linking
        """
        self.threshold = threshold
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        self._initialize_nlp()
        self._setup_filters()

    def _initialize_nlp(self) -> None:
        """Initialize NLP pipeline with UMLS linker"""
        try:
            self.nlp = en_core_sci_md.load()
            if 'scispacy_linker' not in self.nlp.pipe_names:
                self.nlp.add_pipe(
                    "scispacy_linker",
                    config={
                        "resolve_abbreviations": True,
                        "linker_name": "umls",
                        "threshold": self.threshold
                    }
                )
            self.linker = self.nlp.get_pipe("scispacy_linker")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NLP pipeline: {str(e)}")

    def _setup_filters(self) -> None:
        """Setup filters for excluding low-information entities"""
        self.excluded_terms: Set[str] = {
            # Common medical terms
            'procedure', 'treatment', 'therapy', 'intervention',
            'examination', 'observation', 'assessment', 'evaluation',

            # Research related
            'study', 'research', 'analysis', 'investigation',
            'method', 'technique', 'approach', 'protocol',

            # Descriptive terms
            'finding', 'result', 'outcome', 'status', 'condition',
            'state', 'situation', 'problem', 'case', 'type',

            # Time related
            'time', 'duration', 'interval', 'frequency', 'period',
            'onset', 'course', 'history',

            # People related
            'patients', 'person', 'doctor', 'physician', 'specialist',

            # General terms
            'normal', 'abnormal', 'routine', 'standard', 'regular',
            'common', 'typical', 'usual', 'general', 'specific'
        }

    def get_cui_name(self, cui: str) -> Optional[str]:
        """
        Get canonical name for a CUI

        Args:
            cui: UMLS Concept Unique Identifier

        Returns:
            Entity canonical name if found, None otherwise
        """
        try:
            return self.linker.kb.cui_to_entity[cui].canonical_name
        except (KeyError, Exception):
            return None

    def get_name_cui(self, name: str) -> Optional[str]:
        """
        Get CUI for an entity name

        Args:
            name: Entity name to process

        Returns:
            Best matching CUI or None if no match
        """
        if not name or not name.strip():
            return None

        try:
            doc = self.nlp(name.strip())
            best_entity = None
            best_score = 0

            # 找到最佳匹配的实体
            for ent in doc.ents:
                if ent._.kb_ents:
                    cui, score = ent._.kb_ents[0]
                    if self._is_valid_entity(ent, cui, score) and score > best_score:
                        best_entity = (cui, score)
                        best_score = score
            return best_entity[0] if best_entity else None

        except Exception as e:
            return None

    def batch_get_cuis(self, names: List[str], duplicate: bool) -> List[str]:
        """
        Convert multiple names to CUIs

        Args:
            names: List of entity names
            duplicate: is a set or not

        Returns:
            List of unique CUIs found
        """
        all_cuis = []
        for name in names:
            cui = self.get_name_cui(name)
            if cui:
                all_cuis.append(cui)
        return list(set(all_cuis)) if not duplicate else all_cuis  # Remove duplicates

    def batch_get_names(self, cuis: List[str], duplicate: bool) -> List[str]:
        """
        Convert multiple CUIs to names

        Args:
            cuis: List of CUIs

        Returns:
            List of entity names (excluding None values)
        """
        names = []
        for cui in cuis:
            name = self.get_cui_name(cui)
            if name:
                names.append(name)
        return list(set(names)) if not duplicate else names

    def process_text(self, text: str) -> List[str]:
        """
        Process text and extract valid entity CUIs

        Args:
            text: Text to process
            debug: Whether to print debug information
        """
        if not text or not text.strip():
            return []

        try:
            doc = self.nlp(text.strip())
            cuis = []

            for ent in doc.ents:
                if ent._.kb_ents:
                    cui, score = ent._.kb_ents[0]

                    if self._is_valid_entity(ent, cui, score):
                        cuis.append(cui)

            unique_cuis = list(set(cuis))

            return unique_cuis

        except Exception as e:

            return []

    def _is_valid_entity(self, entity, cui: str, score: float, threshold=None) -> bool:
        """
        Check if an entity is valid based on filters
        """
        if threshold is None:
            threshold = self.threshold
        try:
            # 检查分数
            if score < threshold:
                return False

            # 检查实体文本
            if entity.text.lower() in self.excluded_terms:
                return False

            # 检查语义类型
            try:
                entity_info = self.linker.kb.cui_to_entity[cui]
                entity_type = entity_info.types
                semantic_types = {t.split('@')[1] if '@' in t else t for t in entity_type}

                return True

            except KeyError as ke:
                return False

        except Exception as e:
            raise


if __name__ == '__main__':
    cuis = ["C0027750",
            "C0150600",
            "C0043210",
            "C0680063",
            "C0439230",
            "C0032961",
            "C0013080"]
    processor = EntityProcessor()
    text = "Small Round Cell Tumor"
    text1 = 'Neuroblastoma'
    print(processor.batch_get_names(processor.process_text(text), False))
