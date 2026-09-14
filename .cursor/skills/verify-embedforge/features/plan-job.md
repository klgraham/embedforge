# Plan a job

Plan a job is the unpaid paid-operation gate: it shows source revision, embedding settings, output repo, row counts, and estimated cost, and it never calls an embedding API or writes a job.

## Sub-features

- `plan-human` prints the labeled plan plus `No embedding API calls were made.`
- `plan-json` prints source, embedding, output, and estimates as JSON.
- `plan-limit` reduces `estimates.rows` when `--limit` is set.
- `plan-requires-column` rejects a missing `--column`.
- `plan-no-side-effects` leaves `status` empty and the embeddings cache at `Entries: 0`.

## How to get to it (user POV)

- Run `embed plan DATASET --column <column>` in a terminal.
- Add `--limit`, `--split`, `--model`, `--provider`, and `--json` as needed.

## Driving it with control-embed

Preconditions:

- `control-embed doctor` reports version `0.1.0` and a scratch `cache_path`.
- Credential env vars are unset.
- Network is available for a public Hub read of `lhoestq/demo1`.
- `embed status` is `no jobs` before the plan.

- **Require column.** Omit `--column`. Run `control-embed cli -- plan lhoestq/demo1 --limit 2`. Exit code `1`. stderr is `error: --column is required`.
- **Human plan.** Plan two reviews. Run `control-embed cli -- plan lhoestq/demo1 --column review --limit 2 --split train`. Exit code `0`. stdout contains `Dataset:     lhoestq/demo1`, a `Revision:` SHA, `Column:      review`, `Provider:    openai`, `Model:       text-embedding-3-small`, `Dimensions:  1536`, `Repo:        demo1-openai-text-embedding-3-small`, `Rows:        2`, a `Cost:` line, and the banner `No embedding API calls were made.`
- **JSON plan.** Repeat with JSON. Run `control-embed cli -- plan lhoestq/demo1 --column review --limit 2 --split train --json`. Exit code `0`. stdout JSON has `"source": { "repository": "lhoestq/demo1", ... }`, `"embedding": { "provider": "openai", "model": "text-embedding-3-small", "dimensions": 1536, "source_columns": ["review"] }`, `"output": { "repo": "demo1-openai-text-embedding-3-small", "column": "embedding" }`, and `"estimates": { "rows": 2, ... }`.
- **Confirm skipped work.** List jobs and cache. Run `control-embed cli -- status` and `control-embed cli -- cache info`. `status` prints `no jobs`. `cache info` prints `Entries: 0` and a `Path:` under the scratch XDG cache (`.../cache/embedforge/embeddings`).
- **Proof.** Keep the human-plan `.transcript.txt` and the `.side-effects.txt` from that drive. The transcript must include the unpaid banner; the side-effects file must show no job ids and `Entries: 0`.

## Gotchas

- `plan` still reads the Hub to inspect the dataset. That is not a paid embedding call and not an upload. If the Hub is unreachable, report the unmet precondition — do not fall back to calling `build_plan` in Python.
- `--limit` is required for cheap verification. Without it, estimates use the full split.
- `--column` must exist on the dataset. `text` fails on `lhoestq/demo1` (`candidates: id, package_name, review, date`).
- The unpaid banner is necessary but not sufficient. Always re-check `status` and `cache info`.
- `plan` does not create `~/.cache/embedforge/jobs/<id>/`. A new job id means the wrong command (`run`) was used.
- stderr may warn `You are sending unauthenticated requests to the HF Hub`. That is expected without `HF_TOKEN`. OpenAI/OpenRouter errors invalidate the proof.
