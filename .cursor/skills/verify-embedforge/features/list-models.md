# List models

List models shows the offline embedding catalog so a user can pick a provider and model, including default dimensions and whether dimensions are configurable, without touching a network or an API key.

## Sub-features

- `models-table` prints every known model.
- `models-provider` filters the table to `openai` or `openrouter`.
- `models-info` prints details for one model name, including both providers when the short name matches.

## How to get to it (user POV)

- Run `embed models` in a terminal.
- Run `embed models --provider openai` or `embed models --provider openrouter`.
- Run `embed models info <model>` and optionally `--provider <provider>`.

## Driving it with control-embed

Preconditions:

- `control-embed doctor` reports version `0.1.0`.
- No network is required. Credential env vars stay unset.

- **Full table.** List the catalog. Run `control-embed cli -- models`. Exit code `0`. stdout starts with `PROVIDER     MODEL                                DIMENSIONS` and includes `openai       text-embedding-3-small                     1536`, `openai       text-embedding-3-large                     3072`, `openai       text-embedding-ada-002                     1536`, and `openrouter   openai/text-embedding-3-small              1536`.
- **Provider filter.** Restrict to OpenAI. Run `control-embed cli -- models --provider openai`. Exit code `0`. stdout contains the three `openai` rows and does not contain `openrouter`.
- **Info.** Show one short name. Run `control-embed cli -- models info text-embedding-3-small`. Exit code `0`. stdout contains `Model:                   text-embedding-3-small`, `Provider:                openai`, `Dimensions:              1536`, `Configurable dimensions: yes (512–1536)`, `Max input length:        8191 tokens`, and a second block for `openai/text-embedding-3-small` with `Provider:                openrouter`.
- **Unknown model.** Ask for a missing name. Run `control-embed cli -- models info not-a-real-model`. Exit code `1`. stderr contains `error: unknown model 'not-a-real-model'`.
- **Proof.** Capture the full table transcript. The header and the six catalog rows are present. `side-effects.txt` still shows `no jobs` and `Entries: 0`.

## Gotchas

- The catalog is compiled into the CLI. An empty table means the wrong binary, not an empty cache.
- OpenRouter names are prefixed `openai/`. Filtering `--provider openrouter` is required to hide the OpenAI short names.
- `models info text-embedding-3-small` returns two blocks (both providers) unless `--provider` is set.
- This command must not create jobs or cache entries. If `cache info` changes, the instance is contaminated.
