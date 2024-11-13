import spacy
import scispacy
from scispacy.linking import EntityLinker
import en_core_sci_md
import pandas as pd
import warnings
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# 抑制警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class EntityInfo:
    """实体信息数据类"""
    name: str
    types: List[str]
    aliases: List[str]
    definition: Optional[str]
    cui: str
    confidence: float = 0.0


@dataclass
class ProcessedEntityPair:
    """处理后的实体对数据类"""
    original_start: str
    original_end: str
    reasoning: str
    start_cui: Optional[str]
    end_cui: Optional[str]
    start_entity: Optional[EntityInfo]
    end_entity: Optional[EntityInfo]


class MedicalCUIExtractor:
    """医学实体CUI提取器类"""

    def __init__(self, threshold: float = 0.85):
        """初始化NLP模型和实体链接器"""
        self._initialize_nlp(threshold)
        self._setup_filters()

    def _initialize_nlp(self, threshold: float) -> None:
        """初始化NLP模型和链接器"""
        try:
            self.nlp = en_core_sci_md.load()
            if 'scispacy_linker' not in self.nlp.pipe_names:
                self.nlp.add_pipe("scispacy_linker",
                                  config={
                                      "resolve_abbreviations": True,
                                      "linker_name": "umls",
                                      "threshold": threshold
                                  })
            self.linker = self.nlp.get_pipe("scispacy_linker")
        except Exception as e:
            raise RuntimeError(f"NLP模型初始化失败: {str(e)}")

    def _setup_filters(self) -> None:
        """设置实体过滤器"""
        self.blacklist = {
            'source', 'type', 'procedure', 'study', 'treatment', 'patient',
            'following', 'process', 'structure', 'device', 'method'
        }

        self.important_semantic_types = {
            'Disease or Syndrome',
            'Anatomical Structure',
            'Chemical',
            'Clinical Drug',
            'Diagnostic Procedure',
            'Laboratory Procedure',
            'Medical Device',
            'Pharmacologic Substance',
            'Therapeutic or Preventive Procedure'
        }

    def get_cui_name(self, cui: str) -> Optional[str]:
        """
        获取CUI对应的实体名称

        Args:
            cui: UMLS CUI编码

        Returns:
            str: 实体名称，如果未找到返回None
        """
        try:
            entity = self.linker.kb.cui_to_entity[cui]
            return entity.canonical_name if entity else None
        except KeyError:
            return None

    def get_entity_info(self, cui: str) -> Optional[EntityInfo]:
        """获取CUI的详细信息"""
        try:
            entity = self.linker.kb.cui_to_entity[cui]
            return EntityInfo(
                name=entity.canonical_name,
                types=entity.types,
                aliases=entity.aliases,
                definition=entity.definition,
                cui=cui
            )
        except KeyError:
            return None
        except Exception as e:
            print(f"获取实体信息时出错 (CUI: {cui}): {str(e)}")
            return None

    def filter_entities(self, ents) -> List[Tuple[str, float]]:
        """过滤和评分实体"""
        filtered_ents = []
        for ent in ents:
            try:
                if not ent._.kb_ents:
                    continue

                cui, score = ent._.kb_ents[0]
                entity_info = self.get_entity_info(cui)

                if not entity_info or ent.question.lower() in self.blacklist:
                    continue

                # 检查语义类型
                has_important_type = any(t in self.important_semantic_types
                                         for t in entity_info.types)
                if has_important_type:
                    score *= 1.2

                filtered_ents.append((cui, score))
            except Exception as e:
                print(f"实体过滤时出错: {str(e)}")
                continue

        return filtered_ents

    def get_cuis_with_context(self, text: str) -> Dict:
        """从文本中提取CUI列表并包含上下文信息"""
        if pd.isna(text) or not str(text).strip():
            return {'cuis': [], 'entities': []}

        try:
            text = str(text).strip().replace("-", " ")
            doc = self.nlp(text)
            filtered_ents = self.filter_entities(doc.ents)
            filtered_ents.sort(key=lambda x: x[1], reverse=True)

            entities = []
            cuis = []
            for cui, score in filtered_ents:
                entity_info = self.get_entity_info(cui)
                if entity_info:
                    entity_info.confidence = score
                    entities.append(vars(entity_info))
                    cuis.append(cui)

            return {'cuis': cuis, 'entities': entities}
        except Exception as e:
            print(f"处理文本时出错: {str(e)}")
            return {'cuis': [], 'entities': []}

    def process_entity_pair(self, pair: Dict) -> ProcessedEntityPair:
        """处理单个实体对并提取CUI"""
        try:
            start_results = self.get_cuis_with_context(pair['start'])
            end_results = self.get_cuis_with_context(pair['end'])

            return ProcessedEntityPair(
                original_start=pair['start'],
                original_end=pair['end'],
                reasoning=pair.get('reasoning', ''),
                start_cui=start_results['cuis'][0] if start_results['cuis'] else None,
                end_cui=end_results['cuis'][0] if end_results['cuis'] else None,
                start_entity=EntityInfo(**start_results['entities'][0]) if start_results['entities'] else None,
                end_entity=EntityInfo(**end_results['entities'][0]) if end_results['entities'] else None
            )
        except Exception as e:
            print(f"处理实体对时出错: {str(e)}")
            return ProcessedEntityPair(
                original_start=pair['start'],
                original_end=pair['end'],
                reasoning=pair.get('reasoning', ''),
                start_cui=None,
                end_cui=None,
                start_entity=None,
                end_entity=None
            )

