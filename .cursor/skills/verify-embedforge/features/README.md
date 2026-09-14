# EmbedForge verification map

This directory is the maintained source for verifying the user-facing behavior of the EmbedForge `embed` CLI. Read the index before driving the app, then use the matching feature file as the recipe.

## Baseline preconditions

- Linux and macOS are supported. Run `control-embed smoke` on both; do not assume GNU userland or `getent`.
- Launch with `.cursor/skills/verify-embedforge/bin/control-embed launch` so config, jobs, and cache live under `${TMPDIR:-/tmp}/embedforge-verify-$RUN_ID`.
- Isolation uses `HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, and `HF_HOME` on that scratch tree. Leave `EMBEDFORGE_CONFIG_DIR` and `EMBEDFORGE_CACHE_DIR` unset.
- Unset `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`, and `HUGGING_FACE_HUB_TOKEN`.
- Put `control-embed` on `PATH` or invoke it by repo-relative path.
- Run `control-embed doctor` and require version `0.1.0` plus `cache_path` under the scratch XDG cache.
- Never drive an instance that was not started by this verification run.
- Cheap dataset fixture: `lhoestq/demo1`, column `review`, `--split train`. Do not use `embed run` or `embed publish`.

## Driving conventions

- Start every recipe from the baseline state unless its preconditions say otherwise.
- Treat every command as literal. Keep quoted names and flags unchanged.
- Run terminal actions through `control-embed cli -- <embed-args>`.
- Prefer `--json` when asserting structured fields.
- Restore stored config with `embed reset` after a mutation. Do not remove proof artifacts during cleanup.

## Proof and skip reporting

- Capture the user action and the resulting state, not only the final line of stdout.
- CLI proof includes the command, stdout, stderr, and exit code (the `.transcript.txt` file).
- Mutation proof includes a read-only second view (`get`, `config --json`, `status`, or `cache info`).
- After `plan`, also prove the skipped side effects: `status` is `no jobs` and `cache info` shows `Entries: 0`.
- Record the feature ID and entry point used with every artifact.
- Report an unreachable path with the attempted command and the unmet precondition.
- Do not report a skipped entry point as verified through a different path.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with control-embed` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable handles, required state, commands, and observable proof.

## Features

- [Configure settings](./configure.md) covers set/get/unset/reset, `config`, and secret refusal.
- [List models](./list-models.md) covers the offline catalog table and `models info`.
- [Inspect a dataset](./inspect-dataset.md) covers Hub inspect of a public dataset (read, not upload).
- [Plan a job](./plan-job.md) covers the unpaid paid-operation gate with `--limit` and `--json`.
- [Manage the cache](./manage-cache.md) covers `cache info`, `list`, and `clean` on the isolated embeddings cache.
