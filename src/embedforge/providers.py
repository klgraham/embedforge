"""OpenAI and OpenRouter embedding clients."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Never, Protocol

from openai import OpenAI
from openrouter import OpenRouter

from embedforge.errors import EmbedForgeError
from embedforge.shapes import Provider, parse_provider


@dataclass(frozen=True)
class EmbedBatch:
    vectors: list[list[float]]
    tokens: int


class Embedder(Protocol):
    def embed(
        self,
        texts: list[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> EmbedBatch: ...


class OpenAIEmbedder:
    def embed(
        self,
        texts: list[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> EmbedBatch:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EmbedForgeError(
                "API credentials must be supplied through environment variables.\n"
                "Set OPENAI_API_KEY in your shell environment."
            )
        client = OpenAI(api_key=api_key)
        if dimensions is None:
            response = client.embeddings.create(model=model, input=texts)
        else:
            response = client.embeddings.create(model=model, input=texts, dimensions=dimensions)
        return _batch_from_response(response)


class OpenRouterEmbedder:
    def embed(
        self,
        texts: list[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> EmbedBatch:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EmbedForgeError(
                "API credentials must be supplied through environment variables.\n"
                "Set OPENROUTER_API_KEY in your shell environment."
            )
        with OpenRouter(api_key=api_key) as client:
            if dimensions is None:
                response = client.embeddings.generate(model=model, input=texts)
            else:
                response = client.embeddings.generate(
                    model=model, input=texts, dimensions=dimensions
                )
        return _batch_from_response(response)


def get_embedder(provider: str) -> Embedder:
    parsed = parse_provider(provider)
    match parsed:
        case Provider.OPENAI:
            return OpenAIEmbedder()
        case Provider.OPENROUTER:
            return OpenRouterEmbedder()
        case _:
            never: Never = parsed
            raise EmbedForgeError(f"unsupported provider: {never}")


def _batch_from_response(response: object) -> EmbedBatch:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not isinstance(data, list):
        raise EmbedForgeError("embedding response missing data")
    vectors: list[list[float]] = []
    for item in data:
        embedding = getattr(item, "embedding", None)
        if embedding is None and isinstance(item, dict):
            embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise EmbedForgeError("embedding response item missing vector")
        vectors.append([float(value) for value in embedding])
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    tokens = 0
    if usage is not None:
        raw_tokens = getattr(usage, "total_tokens", None)
        if raw_tokens is None and isinstance(usage, dict):
            raw_tokens = usage.get("total_tokens")
        if isinstance(raw_tokens, int) and not isinstance(raw_tokens, bool):
            tokens = raw_tokens
    return EmbedBatch(vectors=vectors, tokens=tokens)
