from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from pathlib import Path
import hashlib
from datetime import datetime


@dataclass
class Question:
    text: str
    options: Dict[str, str]


@dataclass
class EntityLinkingResult:
    question_cuis: List[str]
    option_cuis: Dict[str, List[str]]


@dataclass
class CausalPathResult:
    paths: List[str]


@dataclass
class ReasoningResult:
    reasoning_chain: List[str]
    entity_pairs: List[Dict]


@dataclass
class FinalAnswer:
    answer: str
    confidence: float
    explanation: str


class ProcessingPipeline:
    def __init__(self, config, cache_enabled: bool = True):
        self.config = config
        self.cache_enabled = cache_enabled
        self.cache_dir = config.paths["cache"]

    def _generate_cache_key(self, data: str, prefix: str) -> str:
        """Generate a unique cache key for the input data"""
        hash_object = hashlib.md5(data.encode())
        return f"{prefix}_{hash_object.hexdigest()}"

    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a given key"""
        return self.cache_dir / f"{key}.json"

    def _cache_exists(self, key: str) -> bool:
        """Check if cache exists for the given key"""
        return self._get_cache_path(key).exists()

    def _save_to_cache(self, key: str, data: dict) -> None:
        """Save data to cache"""
        if not self.cache_enabled:
            return

        cache_path = self._get_cache_path(key)
        with cache_path.open('w') as f:
            json.dump({
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'  # 可以用来处理缓存版本兼容性
            }, f)

    def _load_from_cache(self, key: str) -> Optional[dict]:
        """Load data from cache"""
        if not self.cache_enabled:
            return None

        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            with cache_path.open('r') as f:
                cache_data = json.load(f)
                return cache_data['data']
        return None

    def process_question(self, question: Question) -> FinalAnswer:
        """Process a single question through the entire pipeline"""
        # 1. Entity Linking
        entity_result = self._process_entity_linking(question)

        # 2. Causal Path Search
        causal_result = self._process_causal_paths(entity_result)

        # 3. LLM Reasoning
        reasoning_result = self._process_reasoning(question, causal_result)

        # 4. Final Answer Generation
        final_answer = self._generate_final_answer(
            question,
            entity_result,
            causal_result,
            reasoning_result
        )

        return final_answer

    def _process_entity_linking(self, question: Question) -> EntityLinkingResult:
        """Process entity linking with caching"""
        cache_key = self._generate_cache_key(
            f"{question.text}{''.join(question.options.values())}",
            "entity_linking"
        )

        # Try to load from cache
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            return EntityLinkingResult(**cached_result)

        # Process if not in cache
        # ... 实体链接处理逻辑 ...
        result = EntityLinkingResult(...)  # 实际的处理结果

        # Save to cache
        self._save_to_cache(cache_key, {
            'question_cuis': result.question_cuis,
            'option_cuis': result.option_cuis
        })

        return result

    def _process_causal_paths(self, entity_result: EntityLinkingResult) -> CausalPathResult:
        """Process causal path search with caching"""
        # Similar caching logic for causal paths
        ...

    def _process_reasoning(self,
                           question: Question,
                           causal_result: CausalPathResult) -> ReasoningResult:
        """Process LLM reasoning with caching"""
        # Similar caching logic for reasoning
        ...

    def _generate_final_answer(self,
                               question: Question,
                               entity_result: EntityLinkingResult,
                               causal_result: CausalPathResult,
                               reasoning_result: ReasoningResult) -> FinalAnswer:
        """Generate final answer with caching"""
        # Similar caching logic for final answer
        ...

    def process_batch(self, questions: List[Question],
                      output_file: Optional[Path] = None) -> List[FinalAnswer]:
        """Process a batch of questions and optionally save results"""
        results = []
        for question in questions:
            try:
                result = self.process_question(question)
                results.append(result)
            except Exception as e:
                self.config.logger.error(f"Error processing question: {str(e)}")

        if output_file:
            with output_file.open('w') as f:
                json.dump([vars(r) for r in results], f, indent=2)

        return results