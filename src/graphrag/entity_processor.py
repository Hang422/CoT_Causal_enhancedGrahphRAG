import spacy
import scispacy
from scispacy.linking import EntityLinker
import en_core_sci_md
import warnings
from typing import List, Dict, Optional, Set
from src.modules.data_format import EntityInfo, DataFormat, OptionKey, EntityPairsCUIs


class EntityProcessor:
    """Medical entity processor for linking text to UMLS concepts"""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self._initialize_nlp()
        self._setup_semantic_filters()

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

    def get_entity_info(self, cui: str) -> Optional[DataFormat.EntityInfo]:
        """
        Get detailed information for a UMLS CUI

        Args:
            cui: UMLS Concept Unique Identifier

        Returns:
            EntityInfo object if found, None otherwise
        """
        try:
            entity = self.linker.kb.cui_to_entity[cui]
            return DataFormat.EntityInfo(
                name=entity.canonical_name,
                types=entity.types,
                cui=cui
            )
        except (KeyError, Exception) as e:
            return None

    def _setup_semantic_filters(self) -> None:
        """Setup semantic type filters for entity processing"""
        self.excluded_terms: Set[str] = {
            # 一般性医疗术语
            'procedure', 'treatment', 'therapy', 'intervention',
            'examination', 'observation', 'assessment', 'evaluation',
            'test', 'testing', 'screening', 'monitoring',

            # 研究/方法相关
            'study', 'trial', 'research', 'analysis', 'investigation',
            'method', 'technique', 'approach', 'protocol', 'guideline',
            'process', 'procedure', 'measurement', 'evaluation',

            # 一般描述性词语
            'finding', 'result', 'outcome', 'status', 'condition',
            'state', 'situation', 'problem', 'issue', 'case',
            'type', 'form', 'kind', 'category', 'class',
            'level', 'degree', 'stage', 'phase', 'period',

            # 时间/频率相关
            'time', 'duration', 'interval', 'frequency', 'period',
            'onset', 'course', 'history', 'following', 'prior',

            # 人员/角色相关
            'patient', 'subject', 'person', 'individual', 'participant',
            'doctor', 'physician', 'practitioner', 'provider', 'specialist',
            'staff', 'professional', 'clinician', 'researcher',

            # 设施/设备相关
            'facility', 'center', 'unit', 'department', 'clinic',
            'hospital', 'institution', 'device', 'equipment', 'instrument',
            'structure', 'system', 'apparatus',

            # 数量/测量相关
            'amount', 'quantity', 'number', 'value', 'score',
            'rate', 'ratio', 'percentage', 'measure', 'measurement',

            # 一般行为/动作
            'activity', 'action', 'function', 'performance', 'operation',
            'management', 'administration', 'handling', 'control',

            # 位置/方向
            'location', 'site', 'area', 'region', 'position',
            'side', 'part', 'portion', 'section', 'segment',

            # 其他通用词
            'normal', 'abnormal', 'routine', 'standard', 'regular',
            'common', 'typical', 'usual', 'general', 'specific',
            'primary', 'secondary', 'initial', 'final', 'total',
            'complete', 'partial', 'source', 'target', 'base'
        }

        # 低信息量的语义类型代码（Type Unique Identifier - TUI）
        self.excluded_semantic_types: Set[str] = {
            'T033',  # Finding
            'T034',  # Laboratory or Test Result
            'T038',  # Biologic Function
            'T056',  # Daily or Recreational Activity
            'T057',  # Occupational Activity
            'T064',  # Temporal Concept
            'T066',  # Machine Activity
            'T068',  # Human-caused Phenomenon or Process
            'T070',  # Natural Phenomenon or Process
            'T074',  # Medical Device
            'T075',  # Research Device
            'T077',  # Conceptual Entity
            'T079',  # Temporal Concept
            'T080',  # Qualitative Concept
            'T081',  # Quantitative Concept
            'T082',  # Spatial Concept
            'T089',  # Regulation or Law
            'T169',  # Functional Concept
            'T170',  # Intellectual Product
            'T171'  # Language
        }

    def _score_entity(self, entity, base_score: float) -> float:
        """
        Score an entity based on semantic types and other factors

        Args:
            entity: Entity object from spaCy
            base_score: Initial confidence score

        Returns:
            Adjusted confidence score
        """
        try:
            cui = entity._.kb_ents[0][0]
            entity_info = self.get_entity_info(cui)

            if not entity_info:
                return 0.0

            # 检查实体文本是否在黑名单中
            if entity.text.lower() in self.excluded_terms:
                return 0.0

            # 获取语义类型标识符
            semantic_types = set([t.split('@')[1] if '@' in t else t
                                  for t in entity_info.types])

            # 检查是否包含被排除的语义类型
            if any(st in self.excluded_semantic_types for st in semantic_types):
                return 0.0

            return base_score

        except Exception as e:
            return 0.0

    def process_question(self, question: DataFormat.Question) -> DataFormat.QuestionCUIs:
        """
        Process a question to extract and link medical entities

        Args:
            question: Question object containing text and options

        Returns:
            QuestionCUIs containing entities from question and options
        """
        try:
            # Process question text
            question_result = self._process_text(question.text)

            # Process each option
            option_cuis: Dict[OptionKey, List[str]] = {}
            option_entities: Dict[OptionKey, List[DataFormat.EntityInfo]] = {}

            for key, text in question.options.items():
                option_result = self._process_text(text)
                option_cuis[key] = option_result[0]  # CUIs
                option_entities[key] = option_result[1]  # Entities

            return DataFormat.QuestionCUIs(
                question_cuis=question_result[0],
                question_entities=question_result[1],
                option_cuis=option_cuis,
                option_entities=option_entities
            )

        except Exception as e:
            return DataFormat.QuestionCUIs(
                question_cuis=[],
                question_entities=[],
                option_cuis={key: [] for key in OptionKey},
                option_entities={key: [] for key in OptionKey}
            )

    def process_entity_pairs(self, pairs: DataFormat.EntityPairs) -> EntityPairsCUIs:
        """
        Process multiple entity pairs

        Args:
            pairs: List of EntityPairs to process

        Returns:
            List of tuples containing the original pair and its CUIs
        """

        start_cuis = []
        end_cuis = []

        for start_text in pairs.start:
            cuis, _ = self._process_text(start_text)
            start_cuis.extend(cuis)

        for end_text in pairs.end:
            cuis, _ = self._process_text(end_text)
            end_cuis.extend(cuis)

        return EntityPairsCUIs(
            start=start_cuis,
            end=end_cuis,
            reasoning=pairs.reasoning
        )

    def _process_text(self, text: str) -> tuple[List[str], List[DataFormat.EntityInfo]]:
        """
        Process text to extract entities and their CUIs

        Args:
            text: Text to process

        Returns:
            Tuple of (CUI list, EntityInfo list)
        """
        if not text or not text.strip():
            return [], []

        try:
            doc = self.nlp(text.strip())
            entities = []
            cuis = []

            for ent in doc.ents:
                if ent._.kb_ents:
                    cui, score = ent._.kb_ents[0]
                    adjusted_score = self._score_entity(ent, score)

                    if adjusted_score >= self.threshold:
                        entity_info = self.get_entity_info(cui)
                        if entity_info:
                            entity_info.confidence = adjusted_score
                            entities.append(entity_info)
                            cuis.append(cui)

            return cuis, entities

        except Exception:
            return [], []
