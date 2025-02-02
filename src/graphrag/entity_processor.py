import spacy
import scispacy
from scispacy.linking import EntityLinker
import en_core_sci_md
import warnings
from typing import List, Set, Optional


class EntityProcessor:
    """A simple medical entity processor using scispaCy's UMLS linker with extended filters."""

    def __init__(self, threshold: float = 0.9):
        """
        Args:
            threshold: Confidence threshold for entity linking (0~1).
        """
        self.threshold = threshold
        self._setup_filters()
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        self._initialize_nlp()

    def _setup_filters(self) -> None:
        """
        Setup filters for excluding low-information entities based on surface text.

        NOTE: You can add or remove terms to suit your domain needs.
        """
        self.excluded_terms: Set[str] = {
            # Common medical terms or overly broad words
            'procedure', 'treatment', 'therapy', 'intervention',
            'examination', 'observation', 'assessment', 'evaluation',

            # Research related
            'study', 'research', 'analysis', 'investigation',
            'method', 'technique', 'approach', 'protocol',

            # Descriptive or vague terms
            'finding', 'result', 'outcome', 'status', 'condition',
            'state', 'situation', 'problem', 'case', 'type',

            # Time related
            'time', 'duration', 'interval', 'frequency', 'period',
            'onset', 'course', 'history',

            # People related
            'patients', 'patient', 'person', 'doctor', 'physician', 'specialist',

            # General terms
            'normal', 'abnormal', 'routine', 'standard', 'regular',
            'common', 'typical', 'usual', 'general', 'specific',

        }

    def _initialize_nlp(self) -> None:
        """
        Initialize spaCy pipeline with scispaCy UMLS linker.
        """
        self.nlp = en_core_sci_md.load()
        if "scispacy_linker" not in self.nlp.pipe_names:
            self.nlp.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": True,
                    "linker_name": "umls",  # UMLS linker
                    "threshold": self.threshold
                },
            )
        self.linker: EntityLinker = self.nlp.get_pipe("scispacy_linker")

    def _is_low_info_text(self, text: str) -> bool:
        """
        Check if the text is in our excluded list (lowercased).
        """
        return text.lower() in self.excluded_terms

    def _is_valid_entity(self, text: str, cui: str, score: float) -> bool:
        """
        Decide if an extracted entity is valid based on:
          - Not in excluded_terms
          - Linker confidence score >= self.threshold
        """
        if not text or score < self.threshold:
            return False

        # 若文本属于低信息词，直接排除
        if self._is_low_info_text(text):
            return False

        return True

    def get_semantic_types_for_cui(self, cui: str) -> List[str]:
        """
        Return a list of semantic type codes (e.g. T007) for a given CUI.
        If not found, returns an empty list.
        """
        try:
            entity_data = self.linker.kb.cui_to_entity[cui]
        except KeyError:
            return []

        semantic_codes = []
        for full_type_str in entity_data.types:
            # scispaCy 可能返回形如 "T007@Bacterium"
            if "@" in full_type_str:
                type_code, _ = full_type_str.split("@", 1)
                semantic_codes.append(type_code)
            else:
                semantic_codes.append(full_type_str)  # 或者直接放进列表
        return semantic_codes

    def extract_cuis_from_text(self, text: str) -> Set[str]:
        """
        Analyze input text, extract UMLS CUIs for recognized entities
        (filtering by self.threshold and excluded terms).
        Returns a set of unique CUIs.
        """
        if not text or not text.strip():
            return set()
        doc = self.nlp(text)
        found_cuis = set()

        for ent in doc.ents:
            kb_ents = ent._.kb_ents
            if kb_ents:
                top_cui, top_score = kb_ents[0]
                # 检查是否通过过滤
                if self._is_valid_entity(ent.text, top_cui, top_score):
                    found_cuis.add(top_cui)
        return found_cuis


    def extract_semantic_types_from_text(self, text: str) -> Set[str]:
        """
        Extract all CUIs from text, then gather their semantic type codes
        in a set and return it.
        """
        cuis = self.extract_cuis_from_text(text)
        all_type_codes = set()
        for cui in cuis:
            stypes = self.get_semantic_types_for_cui(cui)
            all_type_codes.update(stypes)
        return all_type_codes


if __name__ == "__main__":
    for i in range(3):
        processor = EntityProcessor()
        print(list(processor.extract_cuis_from_text('dopamine production'))[0])


