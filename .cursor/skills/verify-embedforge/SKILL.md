---
name: verify-embedforge
description: Drive the EmbedForge `embed` CLI the way a user does — isolated launch, doctor, feature recipes, and proof artifacts. Use when proving inspect/plan/config/models/cache behavior, checking a CLI change, or verifying unpaid gates without live embedding or Hugging Face uploads.
---

# Verify EmbedForge

Primary surface: the `embed` CLI (`embedforge.cli:main`). There is no web UI, TUI, or long-lived server. Each user command is a short-lived process that prints to stdout/stderr and exits.

Other surfaces (do not treat as the user path): the Python library used by `tests/` (`start_or_resume`, `FakeDatasetSource`, `FakeEmbedder`, `FakeHub`). Those fakes are how unit tests avoid paid APIs. The CLI has no `--test` flag and no env-var that swaps in fakes. Do not invent one.

Read `features/README.md` before driving. Drive the mapped feature you are proving; a single convenient entry point is incomplete when that feature file lists others.

## Platforms

Linux and macOS are supported. Verify launch → doctor → CLI → cleanup on both.

`control-embed` is written for macOS's default Bash 3.2 and BSD userland as well as GNU/Linux. It must not call `getent` on Darwin (stock macOS does not ship it). Login-home lookup uses `dscl` on Darwin, `getent` only when that command exists on other systems, then `~user`. `--help` must succeed when `getent` is absent from `PATH`.

Homebrew `uv` is accepted from `/opt/homebrew/bin/uv` or `/usr/local/bin/uv`. Scratch dirs use `${TMPDIR:-/tmp}` so they follow the platform temp location. Doctor canonicalizes both path sides with Python `os.path.normpath` + `os.path.realpath` so `//` from a trailing `$TMPDIR/` and macOS `/var` vs `/private/var` match.

CI runs `control-embed smoke` on `ubuntu-latest` and `macos-latest`. Locally:

```bash
.cursor/skills/verify-embedforge/bin/control-embed smoke
```

That is unpaid (`models`, `config --json` only): no embedding API calls and no Hub uploads.

## Launch

Install deps once, then start every drive in a fresh isolated env. Ready means `uv run embed --help` prints `usage: embed` and lists `plan`.

```bash
.cursor/skills/verify-embedforge/bin/control-embed launch
```

That command:

1. Runs `uv sync --group dev` from the repo root.
2. Creates a scratch home plus XDG dirs so jobs and cache never land in `~/.cache/embedforge`.
3. Unsets `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `EMBEDFORGE_CONFIG_DIR`, and `EMBEDFORGE_CACHE_DIR`.
4. Prints `control-embed ready run_id=... scratch=... evidence=...`.

Ready check: the printed line is present and `control-embed doctor` later reports `OK`.

Isolation (matches `tests/conftest.py::isolated_home`, plus `HF_HOME` so Hub downloads do not leak):

| Variable | Scratch value |
| --- | --- |
| `HOME` | `/tmp/embedforge-verify-$RUN_ID/home` |
| `XDG_CONFIG_HOME` | `/tmp/embedforge-verify-$RUN_ID/config` |
| `XDG_CACHE_HOME` | `/tmp/embedforge-verify-$RUN_ID/cache` |
| `HF_HOME` | `/tmp/embedforge-verify-$RUN_ID/hf` |

Resolved app paths:

- Config: `$XDG_CONFIG_HOME/embedforge/config.toml`
- Jobs: `$XDG_CACHE_HOME/embedforge/jobs/<id>/`
- Embedding cache: `$XDG_CACHE_HOME/embedforge/embeddings/`

Two verification instances must use different `RUN_ID`s. Each run has its own state file `$CONTROL_EMBED_STATE_DIR/<repo-key>-<run-id>.env`. After `launch`, every later command must select that run with `--run-id` or `CONTROL_EMBED_RUN_ID` (same-process `smoke` keeps the id in memory). Cleanup refuses a scratch directory whose ownership marker does not match this repo and run. Never drive a user's real `~/.cache/embedforge` or `~/.config/embedforge`.

`uv` must be on `PATH` (`https://astral.sh/uv`). If it is missing, install it before launch; that is environment setup, not app behavior.

## Doctor

Read-only. Run first whenever anything looks off, and after every launch.

```bash
.cursor/skills/verify-embedforge/bin/control-embed doctor
```

Pass only when all of these hold:

- `embedforge.__version__` is `0.1.0`
- `uv run embed --help` lists `plan`
- `embed config --json` has `cache_path` equal to `$XDG_CACHE_HOME/embedforge` (the scratch cache, not `$LOGIN_HOME/.cache/embedforge` and not the pre-launch login home cache)
- `config_path` equals `$XDG_CONFIG_HOME/embedforge/config.toml`

Doctor writes `artifacts/<run-id>/doctor.txt` and `doctor-config.json`. If doctor fails, stop; do not drive.

## Drive

Harness: `control-embed` (shipped executable). Every user action is a real `embed` invocation:

```bash
.cursor/skills/verify-embedforge/bin/control-embed cli -- <embed-args>
```

Examples that match this repo (keep flags literal):

```bash
.cursor/skills/verify-embedforge/bin/control-embed cli -- config --json
.cursor/skills/verify-embedforge/bin/control-embed cli -- set provider openai
.cursor/skills/verify-embedforge/bin/control-embed cli -- models --provider openai
.cursor/skills/verify-embedforge/bin/control-embed cli -- inspect lhoestq/demo1 --split train --json
.cursor/skills/verify-embedforge/bin/control-embed cli -- plan lhoestq/demo1 --column review --limit 2 --split train
.cursor/skills/verify-embedforge/bin/control-embed cli -- cache info
.cursor/skills/verify-embedforge/bin/control-embed cli -- status
```

Stable handles are subcommand names, flags, and printed labels — not coordinates:

- Help identity: `usage: embed`
- Plan unpaid banner: `No embedding API calls were made.`
- Plan missing column: `error: --column is required` (exit `1`)
- Secret refusal: `error: API credentials must be supplied through environment variables.` then `Set OPENAI_API_KEY in your shell environment.` (exit `1`)
- Empty jobs: `no jobs`
- Empty cache: `cache is empty`
- Cache info labels: `Path:`, `Entries:`, `Size:`
- Config labels: `Provider:`, `OPENAI_API_KEY:`, `OPENROUTER_API_KEY:`, `HF_TOKEN:`
- JSON config keys: `provider`, `secrets`, `config_path`, `cache_path`

Prefer `--json` when asserting structured output (`config`, `inspect`, `plan`, `status`, `validate`).

Paid and upload commands exist (`run`, `resume`, `publish`) and are out of unpaid verification. Do not call them on the CLI. The library fakes in `tests/fakes.py` are for pytest only. `plan` inspects a public Hub dataset (read); that is not an upload and not an embedding API call. Observe that it skips paid work: no job directory, empty embedding cache, banner printed. Do not trust the name `plan` without those side effects.

Cheap fixture used by the feature map: `lhoestq/demo1`, column `review`, `--split train`, `--limit 2`. README examples use `BeIR/fiqa` and `--column text`; that dataset is larger. Stay on `lhoestq/demo1` unless you are proving `BeIR/fiqa` specifically.

## Evidence

Default location (survives cleanup):

`.cursor/skills/verify-embedforge/artifacts/<run-id>/`

Override with `CONTROL_EMBED_EVIDENCE`. Cleanup deletes scratch only.

Each `cli` drive writes:

- `<seq>-<slug>.cmd.txt` — exact command
- `<seq>-<slug>.stdout.txt` / `.stderr.txt` / `.exit.txt`
- `<seq>-<slug>.transcript.txt` — the `$` line is the shell-quoted `.cmd.txt` contents, then stdout, stderr, exit
- `<seq>-<slug>.side-effects.txt` — `jobs/` listing, `embed cache info`, `embed status`

Proof standards:

- Exercise the real `embed` argv a user would type. Do not call `build_plan(...)` or `main([...])` from Python as a substitute for the CLI path.
- Capture the action and the resulting state (stdout plus jobs/cache/config files), not only the last line.
- After `plan`, require `status` → `no jobs` and `cache info` → `Entries: 0`. After `set`, re-read with `get` or `config --json`.
- Mocks only at production boundaries the app already isolates: Hub inspect is a real read; embedding providers and Hub uploads must not be reached. If stderr contains OpenAI/OpenRouter request errors, the run is invalid.
- Record the feature ID and entry point in the artifact set (copy or name the transcript after the feature id).

## Cleanup

```bash
.cursor/skills/verify-embedforge/bin/control-embed cleanup
```

Selects this run's state, checks the scratch ownership marker (`run_id` + `repo_key`), kills only `ef-verify-$RUN_ID-*` tmux sessions this harness would have created, then `rm -rf` that scratch. It never kills by process name (`embed`, `uv`). It never deletes `$EVIDENCE`. After cleanup, the evidence directory must still exist and still contain the transcripts from the run.

To inspect isolation vars before teardown:

```bash
.cursor/skills/verify-embedforge/bin/control-embed env
```

## Helpers

`bin/control-embed` is executable. Commands:

```bash
.cursor/skills/verify-embedforge/bin/control-embed [--run-id ID] launch
.cursor/skills/verify-embedforge/bin/control-embed [--run-id ID] doctor
.cursor/skills/verify-embedforge/bin/control-embed [--run-id ID] cli -- <embed-args>
.cursor/skills/verify-embedforge/bin/control-embed [--run-id ID] env
.cursor/skills/verify-embedforge/bin/control-embed [--run-id ID] cleanup
.cursor/skills/verify-embedforge/bin/control-embed [--run-id ID] smoke
```

Optional env: `CONTROL_EMBED_RUN_ID`, `CONTROL_EMBED_SCRATCH`, `CONTROL_EMBED_EVIDENCE`, `CONTROL_EMBED_STATE`, `CONTROL_EMBED_STATE_DIR`.

## Guardrails

- No paid embedding API calls. No `embed run` / `embed resume` / `embed publish` on the live CLI.
- No live Hugging Face uploads. `inspect` and `plan` may read a public dataset.
- Always pass `--limit` on `plan` (and on `run` if a later paid run is ever explicitly authorized).
- Unset credential env vars for unpaid proofs. `embed set` must refuse keys; never write secrets into config.toml.
- Do not double-drive a shared instance. If doctor reports the user cache path, refuse.

When the app changes, keep this map honest with `/maintain-verification-skill`.
