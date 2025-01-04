from typing import Any
from neo4j_graphrag.embeddings.base import Embedder

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


class CohereEmbeddings(Embedder):
    def __init__(self, model: str = "", **kwargs: Any) -> None:
        if cohere is None:
            raise ImportError(
                "Could not import cohere python client. "
                "Please install it with `pip install cohere`."
            )
        self.model = model
        self.client = cohere.Client(**kwargs)

    def embed_query(self, text: str, input_type: str = "search_query", **kwargs: Any) -> list[float]:
        kwargs.setdefault("truncate", "END")  # 确保一致性
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type=input_type,  # 必须指定
            **kwargs,
        )
        return response.embeddings[0]