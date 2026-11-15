"""Turn text into vectors. Two implementations behind one interface.

FastEmbedEmbedder is the real one: a small open model that runs locally (no API key, no cost).
HashingEmbedder is a deterministic stand-in for tests: no network, no download, no meaning.
"""

import hashlib
import math
from typing import Protocol

DIMENSIONS = 384  # must match vector(384) in sql/001_catalogue.sql


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # imported here so tests don't need the model

        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models embed search queries slightly differently from the documents they search
        return next(iter(self._model.query_embed([text]))).tolist()


class HashingEmbedder:
    """Character trigrams hashed into buckets. Similar spelling gives similar vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        padded = f"  {text.lower()}  "
        for i in range(len(padded) - 2):
            bucket = int(hashlib.md5(padded[i : i + 3].encode()).hexdigest(), 16) % DIMENSIONS
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
