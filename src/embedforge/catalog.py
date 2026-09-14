"""Offline embedding-model catalog, pricing, and token estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass

from embedforge.errors import EmbedForgeError
from embedforge.shapes import Provider, parse_provider

CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class ModelInfo:
    provider: Provider
    model: str
    dimensions: int
    configurable_dimensions: bool
    min_dimensions: int | None
    max_input_tokens: int
    usd_per_million_tokens: float

    @property
    def slug(self) -> str:
        return self.model

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider.value,
            "model": self.model,
            "dimensions": self.dimensions,
            "configurable_dimensions": self.configurable_dimensions,
            "min_dimensions": self.min_dimensions,
            "max_input_tokens": self.max_input_tokens,
            "usd_per_million_tokens": self.usd_per_million_tokens,
        }


def _openai(
    model: str,
    dimensions: int,
    *,
    configurable: bool,
    min_dimensions: int | None,
    price: float,
) -> ModelInfo:
    return ModelInfo(
        provider=Provider.OPENAI,
        model=model,
        dimensions=dimensions,
        configurable_dimensions=configurable,
        min_dimensions=min_dimensions,
        max_input_tokens=8191,
        usd_per_million_tokens=price,
    )


def _openrouter(model: str, base: ModelInfo) -> ModelInfo:
    return ModelInfo(
        provider=Provider.OPENROUTER,
        model=model,
        dimensions=base.dimensions,
        configurable_dimensions=base.configurable_dimensions,
        min_dimensions=base.min_dimensions,
        max_input_tokens=base.max_input_tokens,
        usd_per_million_tokens=base.usd_per_million_tokens,
    )


_OPENAI_MODELS = (
    _openai(
        "text-embedding-3-small",
        1536,
        configurable=True,
        min_dimensions=512,
        price=0.02,
    ),
    _openai(
        "text-embedding-3-large",
        3072,
        configurable=True,
        min_dimensions=256,
        price=0.13,
    ),
    _openai(
        "text-embedding-ada-002",
        1536,
        configurable=False,
        min_dimensions=None,
        price=0.10,
    ),
)

MODELS: tuple[ModelInfo, ...] = _OPENAI_MODELS + tuple(
    _openrouter(f"openai/{item.model}", item) for item in _OPENAI_MODELS
)


def list_models(provider: str | None = None) -> tuple[ModelInfo, ...]:
    if provider is None:
        return MODELS
    wanted = parse_provider(provider)
    return tuple(item for item in MODELS if item.provider is wanted)


def find_models(model: str, provider: str | None = None) -> tuple[ModelInfo, ...]:
    wanted = parse_provider(provider) if provider else None
    needle = model.strip()
    matches = [
        item for item in MODELS if item.model == needle or item.model.split("/")[-1] == needle
    ]
    if wanted is not None:
        matches = [item for item in matches if item.provider is wanted]
    return tuple(matches)


def resolve_model(
    provider: str,
    model: str,
    dimensions: int | None,
) -> ModelInfo:
    parsed = parse_provider(provider)
    resolved_name = model
    if parsed is Provider.OPENROUTER and "/" not in model:
        resolved_name = f"openai/{model}"
    matches = find_models(resolved_name, provider=parsed.value)
    if not matches:
        matches = find_models(model, provider=parsed.value)
    if not matches:
        return ModelInfo(
            provider=parsed,
            model=resolved_name,
            dimensions=dimensions or 1536,
            configurable_dimensions=dimensions is not None,
            min_dimensions=None,
            max_input_tokens=8191,
            usd_per_million_tokens=_price_for_name(resolved_name),
        )
    info = matches[0]
    if dimensions is None:
        return info
    if not info.configurable_dimensions and dimensions != info.dimensions:
        raise EmbedForgeError(
            f"{info.model} does not support configurable dimensions (fixed at {info.dimensions})"
        )
    if info.min_dimensions is not None and dimensions < info.min_dimensions:
        raise EmbedForgeError(f"dimensions must be >= {info.min_dimensions} for {info.model}")
    if dimensions > info.dimensions:
        raise EmbedForgeError(f"dimensions must be <= {info.dimensions} for {info.model}")
    return ModelInfo(
        provider=info.provider,
        model=info.model,
        dimensions=dimensions,
        configurable_dimensions=info.configurable_dimensions,
        min_dimensions=info.min_dimensions,
        max_input_tokens=info.max_input_tokens,
        usd_per_million_tokens=info.usd_per_million_tokens,
    )


def estimate_tokens(characters: int) -> int:
    if characters <= 0:
        return 0
    return math.ceil(characters / CHARS_PER_TOKEN)


def estimate_cost_usd(tokens: int, usd_per_million: float) -> float:
    return tokens * usd_per_million / 1_000_000


def _price_for_name(model: str) -> float:
    suffix = model.split("/")[-1]
    for item in _OPENAI_MODELS:
        if item.model == suffix:
            return item.usd_per_million_tokens
    return 0.02
