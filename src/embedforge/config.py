"""Load and store non-secret EmbedForge settings."""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from embedforge.errors import EmbedForgeError, SecretInConfigError
from embedforge.paths import config_path
from embedforge.shapes import (
    ALLOWED_CONFIG_KEYS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_COLUMN,
    DEFAULT_PROVIDER,
    DEFAULT_STORAGE_DTYPE,
    Config,
    parse_provider,
    parse_storage_dtype,
)

_SECRET_ENV_BY_KEY = {
    "openai_api_key": "OPENAI_API_KEY",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "hf_token": "HF_TOKEN",
    "huggingface_token": "HF_TOKEN",
    "hugging_face_hub_token": "HF_TOKEN",
    "api_key": "OPENAI_API_KEY",
    "token": "HF_TOKEN",
    "secret": "OPENAI_API_KEY",
    "password": "OPENAI_API_KEY",
    "credential": "OPENAI_API_KEY",
}

_SECRET_SUBSTRINGS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
)

_INT_KEYS = {"batch_size", "concurrency", "dimensions"}


def secret_env_status() -> dict[str, bool]:
    return {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
        "HF_TOKEN": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    }


def refuse_if_secret(key: str) -> None:
    env_name = secret_env_for_key(key)
    if env_name is not None:
        raise SecretInConfigError(env_name)


def secret_env_for_key(key: str) -> str | None:
    normalized = _normalize_key(key)
    if normalized in _SECRET_ENV_BY_KEY:
        return _SECRET_ENV_BY_KEY[normalized]
    compact = normalized.replace("_", "")
    for fragment in _SECRET_SUBSTRINGS:
        if fragment in normalized or fragment.replace("_", "") in compact:
            if "openrouter" in normalized:
                return "OPENROUTER_API_KEY"
            if "hf" in normalized or "hugging" in normalized:
                return "HF_TOKEN"
            if "openai" in normalized or fragment in {"api_key", "apikey", "secret"}:
                return "OPENAI_API_KEY"
            return "OPENAI_API_KEY"
    return None


def load_config() -> Config:
    stored = read_config_file()
    return resolve_config(stored)


def resolve_config(stored: dict[str, Any]) -> Config:
    provider = stored.get("provider", DEFAULT_PROVIDER)
    if not isinstance(provider, str):
        raise EmbedForgeError("provider must be a string")
    parse_provider(provider)
    model = stored.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or not model:
        raise EmbedForgeError("model must be a non-empty string")
    output_column = stored.get("output_column", DEFAULT_OUTPUT_COLUMN)
    if not isinstance(output_column, str) or not output_column:
        raise EmbedForgeError("output_column must be a non-empty string")
    namespace = stored.get("hf_namespace")
    if namespace is not None and not isinstance(namespace, str):
        raise EmbedForgeError("hf_namespace must be a string")
    return Config(
        provider=parse_provider(provider).value,
        model=model,
        batch_size=_as_positive_int(stored.get("batch_size", DEFAULT_BATCH_SIZE), "batch_size"),
        concurrency=_as_positive_int(stored.get("concurrency", DEFAULT_CONCURRENCY), "concurrency"),
        dimensions=_optional_positive_int(stored.get("dimensions"), "dimensions"),
        output_column=output_column,
        storage_dtype=parse_storage_dtype(
            _string_value(stored.get("storage_dtype", DEFAULT_STORAGE_DTYPE), "storage_dtype")
        ).value,
        hf_namespace=namespace or None,
    )


def get_value(key: str) -> str:
    refuse_if_secret(key)
    normalized = _normalize_config_key(key)
    config = load_config()
    value = getattr(config, normalized)
    if value is None:
        return "not set"
    return str(value)


def set_value(key: str, raw: str) -> Config:
    refuse_if_secret(key)
    normalized = _normalize_config_key(key)
    stored = read_config_file()
    stored[normalized] = parse_config_value(normalized, raw)
    write_config_file(stored)
    return resolve_config(stored)


def unset_value(key: str) -> Config:
    refuse_if_secret(key)
    normalized = _normalize_config_key(key)
    stored = read_config_file()
    stored.pop(normalized, None)
    write_config_file(stored)
    return resolve_config(stored)


def reset_config() -> None:
    path = config_path()
    if path.exists():
        path.unlink()


def read_config_file() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    return parse_simple_toml(path.read_text())


def write_config_file(stored: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_simple_toml(stored))


def parse_config_value(key: str, raw: str) -> str | int:
    if key in _INT_KEYS:
        return _as_positive_int(raw, key)
    if key == "provider":
        return parse_provider(raw).value
    if key == "storage_dtype":
        return parse_storage_dtype(raw).value
    if not raw.strip():
        raise EmbedForgeError(f"{key} must be a non-empty string")
    return raw.strip()


def apply_overrides(
    config: Config,
    *,
    provider: str | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    output_column: str | None = None,
    batch_size: int | None = None,
    concurrency: int | None = None,
    storage_dtype: str | None = None,
    hf_namespace: str | None = None,
) -> Config:
    updated = config
    if provider is not None:
        updated = replace(updated, provider=parse_provider(provider).value)
    if model is not None:
        updated = replace(updated, model=model)
    if dimensions is not None:
        updated = replace(updated, dimensions=_as_positive_int(dimensions, "dimensions"))
    if output_column is not None:
        updated = replace(updated, output_column=output_column)
    if batch_size is not None:
        updated = replace(updated, batch_size=_as_positive_int(batch_size, "batch_size"))
    if concurrency is not None:
        updated = replace(updated, concurrency=_as_positive_int(concurrency, "concurrency"))
    if storage_dtype is not None:
        updated = replace(updated, storage_dtype=parse_storage_dtype(storage_dtype).value)
    if hf_namespace is not None:
        updated = replace(updated, hf_namespace=hf_namespace)
    return updated


def parse_simple_toml(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise EmbedForgeError(f"invalid config line {line_no}")
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if key not in ALLOWED_CONFIG_KEYS:
            raise EmbedForgeError(f"unknown config key {key!r}")
        parsed[key] = _parse_toml_value(raw_value.strip(), key)
    return parsed


def dump_simple_toml(stored: dict[str, Any]) -> str:
    lines: list[str] = []
    for key in ALLOWED_CONFIG_KEYS:
        if key not in stored or stored[key] is None:
            continue
        value = stored[key]
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
        elif isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        else:
            lines.append(f"{key} = {value}")
    return ("\n".join(lines) + "\n") if lines else ""


def _parse_toml_value(raw: str, key: str) -> str | int:
    if raw in {"null", "None"}:
        raise EmbedForgeError(f"{key} cannot be null in config")
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    if re.fullmatch(r"-?\d+", raw):
        return _as_positive_int(int(raw), key) if key in _INT_KEYS else int(raw)
    return parse_config_value(key, raw)


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _normalize_config_key(key: str) -> str:
    normalized = _normalize_key(key)
    if normalized not in ALLOWED_CONFIG_KEYS:
        allowed = ", ".join(ALLOWED_CONFIG_KEYS)
        raise EmbedForgeError(f"unknown setting {key!r}; expected one of: {allowed}")
    return normalized


def _as_positive_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise EmbedForgeError(f"{key} must be a positive integer")
    if isinstance(value, str):
        if not re.fullmatch(r"-?\d+", value.strip()):
            raise EmbedForgeError(f"{key} must be a positive integer")
        number = int(value.strip())
    else:
        number = value
    if number <= 0:
        raise EmbedForgeError(f"{key} must be a positive integer")
    return number


def _optional_positive_int(value: object, key: str) -> int | None:
    if value is None:
        return None
    return _as_positive_int(value, key)


def _string_value(value: object, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise EmbedForgeError(f"{key} must be a non-empty string")
    return value


def format_secret_status(available: bool) -> str:
    return "✓ available" if available else "not set"


def config_dir_display(path: Path | None = None) -> str:
    return str(path or config_path())
