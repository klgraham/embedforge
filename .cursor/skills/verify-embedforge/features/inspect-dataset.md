# Inspect a dataset

Inspect a dataset shows the Hub repository, revision SHA, license, configs, splits, columns, candidate text columns, and estimated embedding input so a user can choose `--column` before planning. It reads public dataset metadata; it does not embed and does not upload.

## Sub-features

- `inspect-human` prints the labeled report.
- `inspect-json` prints the same fields as JSON.
- `inspect-split` honors `--split` and `--config`.

## How to get to it (user POV)

- Run `embed inspect DATASET` in a terminal.
- Run `embed inspect DATASET --split train --config default --json`.

## Driving it with control-embed

Preconditions:

- `control-embed doctor` reports the scratch `cache_path`.
- Network is available to `huggingface.co` for a public read.
- Credential env vars stay unset. Do not pass `HF_TOKEN`.
- Use `lhoestq/demo1` unless proving another dataset.

- **Human inspect.** Inspect the fixture. Run `control-embed cli -- inspect lhoestq/demo1 --split train`. Exit code `0`. stdout contains `Repository:   lhoestq/demo1`, a `Revision:` value that is a 40-character hex SHA, `Split:        train`, and `Text columns:` including `review`.
- **JSON inspect.** Repeat with JSON. Run `control-embed cli -- inspect lhoestq/demo1 --split train --json`. Exit code `0`. stdout JSON has `"repository": "lhoestq/demo1"`, a non-empty `"revision"`, `"split": "train"`, and `"candidate_text_columns"` containing `review`.
- **Proof.** Capture the JSON transcript. `side-effects.txt` still shows `no jobs` and `Entries: 0`. stderr may contain `You are sending unauthenticated requests to the HF Hub`; that warning is not a failure. stderr must not mention OpenAI or OpenRouter.

## Gotchas

- This is a Hub **read**. A 401/403 or DNS failure is an unmet network precondition, not an app-logic fail. Record the command and stop.
- Column names are dataset-specific. `lhoestq/demo1` uses `review`, not `text`. `BeIR/fiqa` (README example) is larger and slower.
- Inspect does not accept `--limit`. Size control belongs on `plan` / `run`.
- Do not treat a Hub download cache under `HF_HOME` as an EmbedForge job. Jobs only appear under `$XDG_CACHE_HOME/embedforge/jobs`.
- Never follow inspect with `embed run` during unpaid verification.
