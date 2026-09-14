# Configure settings

Configure settings lets a user store non-secret defaults, read them back, clear them, and see whether API credentials exist in the environment — without ever writing a secret to disk.

## Sub-features

- `config-show` prints human settings and secret availability.
- `config-json` prints the same fields as JSON, including `config_path` and `cache_path`.
- `config-set` stores an allowed key (`provider`, `model`, `batch_size`, `concurrency`, `dimensions`, `output_column`, `hf_namespace`).
- `config-get` prints one stored or default value.
- `config-unset` removes one stored key so the default returns.
- `config-reset` deletes the config file and restores defaults.
- `config-refuse-secret` rejects `openai_api_key` and similar keys with the env-var error.

## How to get to it (user POV)

- Run `embed config` or `embed config --json` in a terminal.
- Run `embed set <key> <value>`, `embed get <key>`, `embed unset <key>`, or `embed reset`.

## Driving it with control-embed

Preconditions:

- `control-embed doctor` reports version `0.1.0` and a scratch `cache_path`.
- Credential env vars are unset.
- No paid command will be run.

- **Show defaults.** Print JSON config. Run `control-embed cli -- config --json`. Exit code `0`. stdout includes `"provider": "openai"`, `"model": "text-embedding-3-small"`, `"batch_size": 128`, `"concurrency": 8`, `"secrets": { "OPENAI_API_KEY": false, "OPENROUTER_API_KEY": false, "HF_TOKEN": false }`, and a `cache_path` under the scratch XDG cache.
- **Human view.** Print the labeled listing. Run `control-embed cli -- config`. stdout contains `Provider:        openai` and `OPENAI_API_KEY:  not set`.
- **Set a value.** Store a non-secret. Run `control-embed cli -- set provider openrouter`. Exit code `0` and stdout is `provider = openrouter`.
- **Read it back.** Run `control-embed cli -- get provider`. Exit code `0` and stdout is `openrouter`.
- **Refuse a secret.** Attempt to store a key. Run `control-embed cli -- set openai_api_key sk-secret`. Exit code `1`. stderr contains `error: API credentials must be supplied through environment variables.` and `Set OPENAI_API_KEY in your shell environment.` The config file under `config_path` does not contain `sk-secret`.
- **Unset.** Run `control-embed cli -- unset model` then `control-embed cli -- get model`. After unset, `get model` prints `text-embedding-3-small`.
- **Reset.** Run `control-embed cli -- reset` then `control-embed cli -- get provider`. stdout of reset is `config reset`. `get provider` is `openai`.
- **Proof.** Re-read JSON after reset. Run `control-embed cli -- config --json`. Capture `.transcript.txt`. `provider` is `openai` and `cache_path` is still the scratch path.

## Gotchas

- `embed set` argument order is `key` then `value`. There is no `embed set --provider`.
- Allowed keys are only `provider`, `model`, `batch_size`, `concurrency`, `dimensions`, `output_column`, `hf_namespace`. Unknown keys fail.
- Secret detection matches substrings (`api_key`, `token`, `secret`). A refused key is not proof that config is healthy — read `config --json` afterward.
- `config --json` may report a secret as available if the agent leaked an env var into the process. Doctor unsets the known credential names; do not export them for unpaid proofs.
- Defaults exist even with no config file. `unset` / `reset` restore defaults; they do not print empty strings.
