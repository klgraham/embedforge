# Manage the cache

Manage the cache lets a user see where embeddings are stored, list cache keys, and delete cached vectors. On a clean unpaid instance the cache is empty because `plan` and `inspect` do not write embeddings.

## Sub-features

- `cache-info` prints path, entry count, and size.
- `cache-list` prints keys or `cache is empty`.
- `cache-clean` deletes cached embeddings and notes that `embed cache gc` is not in V1.
- `cache-help` explains the subcommands when `cache` is invoked alone.

## How to get to it (user POV)

- Run `embed cache info`, `embed cache list`, or `embed cache clean` in a terminal.
- Run `embed cache` with no subcommand to see usage.

## Driving it with control-embed

Preconditions:

- `control-embed doctor` reports a scratch `cache_path`.
- No `embed run` has been executed in this instance.
- Credential env vars stay unset.

- **Bare cache.** Omit the subcommand. Run `control-embed cli -- cache`. Exit code `1`. stderr is `error: usage: embed cache {info,list,clean}`.
- **Info on empty cache.** Run `control-embed cli -- cache info`. Exit code `0`. stdout contains `Path:    ` ending in `/cache/embedforge/embeddings`, `Entries: 0`, and `Size:    0 B`. The path is under the scratch XDG cache, not `~/.cache/embedforge`.
- **List empty.** Run `control-embed cli -- cache list`. Exit code `0`. stdout is `cache is empty`.
- **Clean empty.** Run `control-embed cli -- cache clean`. Exit code `0`. stdout contains `removed 0 cached embeddings` and `note: embed cache gc is planned for a later release`.
- **Proof.** Capture `cache info` transcript. Path stays inside the scratch tree. `status` remains `no jobs`.

## Gotchas

- There is no `embed cache gc` in V1. Asking for it is a usage error, not a clean.
- `Path` is the embeddings cache, not the jobs directory. Jobs live in `.../embedforge/jobs/`.
- An unpaid `plan` must not increment `Entries`. If it does, stop and treat the instance as contaminated.
- `cache clean` deletes embedding files only. It does not remove proof artifacts or the evidence directory.
- Filling the cache requires `embed run` (paid or library fakes). Do not do that in unpaid CLI verification.
