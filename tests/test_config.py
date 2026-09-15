from __future__ import annotations

import json
from pathlib import Path

from embedforge.cli import main
from embedforge.config import load_config, secret_env_for_key
from embedforge.errors import SecretInConfigError
from embedforge.paths import config_path


def test_set_refuses_openai_api_key(isolated_home, capsys) -> None:
    code = main(["set", "openai_api_key", "sk-secret"])
    captured = capsys.readouterr()
    assert code == 1
    assert "error: API credentials must be supplied through environment variables." in captured.err
    assert "Set OPENAI_API_KEY in your shell environment." in captured.err
    assert load_config().provider == "openai"
    if config_path().exists():
        assert "sk-secret" not in config_path().read_text()
    for path in Path(isolated_home).rglob("*"):
        if path.is_file():
            assert "sk-secret" not in path.read_text()


def test_set_refuses_hf_token_and_generic_api_key(isolated_home, capsys) -> None:
    assert main(["set", "hf_token", "hf_xxx"]) == 1
    err = capsys.readouterr().err
    assert "Set HF_TOKEN in your shell environment." in err
    assert main(["set", "api_key", "sk-xxx"]) == 1
    err = capsys.readouterr().err
    assert "Set OPENAI_API_KEY in your shell environment." in err


def test_secret_detection_covers_openrouter() -> None:
    assert secret_env_for_key("OPENROUTER_API_KEY") == "OPENROUTER_API_KEY"
    assert secret_env_for_key("openrouter-api-key") == "OPENROUTER_API_KEY"


def test_set_get_unset_reset_roundtrip(isolated_home, capsys) -> None:
    assert main(["set", "provider", "openrouter"]) == 0
    assert main(["set", "model", "text-embedding-3-large"]) == 0
    assert main(["set", "batch_size", "16"]) == 0
    assert main(["set", "storage_dtype", "float16"]) == 0
    capsys.readouterr()
    assert main(["get", "provider"]) == 0
    assert capsys.readouterr().out.strip() == "openrouter"
    assert main(["get", "storage_dtype"]) == 0
    assert capsys.readouterr().out.strip() == "float16"
    assert main(["unset", "model"]) == 0
    capsys.readouterr()
    assert main(["get", "model"]) == 0
    assert capsys.readouterr().out.strip() == "text-embedding-3-small"
    assert main(["reset"]) == 0
    capsys.readouterr()
    assert main(["get", "provider"]) == 0
    assert capsys.readouterr().out.strip() == "openai"


def test_config_json_never_prints_secret_values(isolated_home, capsys, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-secret-value")
    monkeypatch.setenv("HF_TOKEN", "hf_live_secret")
    assert main(["config", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    dumped = json.dumps(payload)
    assert "sk-live-secret-value" not in dumped
    assert "hf_live_secret" not in dumped
    assert payload["secrets"]["OPENAI_API_KEY"] is True
    assert payload["secrets"]["OPENROUTER_API_KEY"] is False
    assert payload["secrets"]["HF_TOKEN"] is True


def test_direct_refuse_if_secret_matches_spec_text() -> None:
    try:
        from embedforge.config import refuse_if_secret

        refuse_if_secret("openai_api_key")
    except SecretInConfigError as exc:
        assert str(exc) == (
            "API credentials must be supplied through environment variables.\n"
            "Set OPENAI_API_KEY in your shell environment."
        )
    else:
        raise AssertionError("expected SecretInConfigError")
