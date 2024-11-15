import spacy
import scispacy
from scispacy.linking import EntityLinker
import en_core_sci_md
import pandas as pd
import warnings
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class EntityInfo:
    """Entity information data class"""
    name: str
    types: List[str]
    aliases: List[str]
    definition: Optional[str]
    cui: str
    confidence: float = 0.0


@dataclass
class ProcessedEntityPair:
    """Processed entity pair data class"""
    original_start: str
    original_end: str
    reasoning: str
    start_cui: Optional[str]
    end_cui: Optional[str]
    start_entity: Optional[EntityInfo]
    end_entity: Optional[EntityInfo]


class MedicalCUIExtractor:
    """Medical Entity CUI Extractor"""

    def __init__(self, threshold: float = 0.85):
        """Initialize the NLP model and entity linker"""
        self._initialize_nlp(threshold)
        self._setup_filters()

    def _initialize_nlp(self, threshold: float) -> None:
        """Initialize the NLP model and linker"""
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
            raise RuntimeError(f"NLP model initialization failed: {str(e)}")

    def _setup_filters(self) -> None:
        """Set up entity filters"""
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
        Get the entity name corresponding to the CUI

        Args:
            cui: UMLS CUI code

        Returns:
            str: Entity name, or None if not found
        """
        try:
            entity = self.linker.kb.cui_to_entity[cui]
            return entity.canonical_name if entity else None
        except KeyError:
            return None

    def get_entity_info(self, cui: str) -> Optional[EntityInfo]:
        """Get detailed information for a CUI"""
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
            print(f"Error retrieving entity info (CUI: {cui}): {str(e)}")
            return None

    def filter_entities(self, ents) -> List[Tuple[str, float]]:
        """Filter and score entities"""
        filtered_ents = []
        for ent in ents:
            try:
                if not ent._.kb_ents:
                    continue

                cui, score = ent._.kb_ents[0]
                entity_info = self.get_entity_info(cui)

                if not entity_info or ent.text.lower() in self.blacklist:
                    continue

                # Check semantic types
                has_important_type = any(t in self.important_semantic_types
                                         for t in entity_info.types)
                if has_important_type:
                    score *= 1.2

                filtered_ents.append((cui, score))
            except Exception as e:
                print(f"Error during entity filtering: {str(e)}")
                continue

        return filtered_ents

    def get_cuis_with_context(self, text: str) -> Dict:
        """Extract a list of CUIs from text along with context information"""
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
            print(f"Error processing text: {str(e)}")
            return {'cuis': [], 'entities': []}

    def process_entity_pair(self, pair: Dict) -> ProcessedEntityPair:
        """Process a single entity pair and extract CUIs"""
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
            print(f"Error processing entity pair: {str(e)}")
            return ProcessedEntityPair(
                original_start=pair['start'],
                original_end=pair['end'],
                reasoning=pair.get('reasoning', ''),
                start_cui=None,
                end_cui=None,
                start_entity=None,
                end_entity=None
            )


if __name__ == '__main__':
    text = "Which of the following hormone is/are under inhibitory of hypothalamus?"
    processor = MedicalCUIExtractor()
    results = processor.get_cuis_with_context(text)
    print("Extracted CUIs:", results['cuis'])
    print("Entities:")
    for entity in results['entities']:
        print(entity)