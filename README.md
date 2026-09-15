# EmbedForge

CLI that inspects a Hugging Face dataset, plans and runs embedding jobs through
OpenAI or OpenRouter, validates the result, and publishes a derived dataset with
exact source-revision provenance.

```bash
uv sync
uv run embed --help
```

Install from a checkout:

```bash
uv tool install .
embed inspect BeIR/fiqa
```

## Pipeline

Commands stay small and composable. There is no one-shot `embed dataset` command.

```text
embed inspect DATASET
embed set provider openai
embed set model text-embedding-3-small
embed set storage_dtype float32
embed plan DATASET --column text
embed run DATASET --column text --limit 32
embed status
embed resume JOB_ID
embed validate JOB_ID
embed publish JOB_ID
```

`plan` is a first-class paid-operation gate. It shows the source revision, output
shape, destination, and estimated cost, and it never calls an embedding API.

The default output repository name is
`{dataset}-{provider}-{model}-{dimensions}`, for example
`fiqa-openai-text-embedding-3-small-1536`. Provider-prefixed model names are
normalized (`openai/text-embedding-3-small` becomes `text-embedding-3-small`).
`run` records that destination on the local job; `publish` uses it unless you
pass `--repo owner/name`.

EmbedForge stores each embedding as a fixed-size vector. The default `float32`
format uses half the space of an inferred Python `float64` list. Use
`--storage-dtype float16` or `embed set storage_dtype float16` to halve the
vector payload again. `float16` changes the stored values, so measure retrieval
quality before using it for a published dataset. Hugging Face publication uses
Parquet; local staged datasets use memory-mapped Arrow.

`run` writes a local job under `~/.cache/embedforge/jobs/<id>/` and does not
publish. `publish` defaults to a **private** Hugging Face dataset.

If the selected source column contains an empty string or `null`, `run` keeps
the row and writes `null` to the embedding column. Validation allows that null
only for an empty source value. EmbedForge does not write a zero vector because
a zero vector would look like an embedding to downstream consumers.

## Configuration

```bash
embed set provider openai
embed set model text-embedding-3-small
embed set batch_size 128
embed set concurrency 8
embed set storage_dtype float32
embed get provider
embed config
embed config --json
embed unset model
embed reset
```

Secrets are environment variables only. `embed set` refuses API keys and tokens.
Config files never store credentials.

```bash
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export HF_TOKEN=...
```

`embed config` prints settings and whether those secrets exist, never their values.

## Models, cache, and jobs

```bash
embed models
embed models --provider openai
embed models info text-embedding-3-small
embed cache info
embed cache list
embed cache clean
embed storage migrate
```

Embedding cache keys are `H(provider, model, dimensions, input)` so retries do
not re-pay identical values. The cache stores one packed `float32` vector per
key in `~/.cache/embedforge/embeddings/embeddings.sqlite3`. Job journals record
row status and cache keys without copying vectors into JSON. Jobs use ULID-style
ids and are resumable with `embed resume JOB_ID` or `embed run --resume JOB_ID`.

To convert caches and journals created by an older EmbedForge version, run:

```bash
embed storage migrate
embed storage migrate --delete-legacy-cache
```

The first command copies legacy JSON cache entries into SQLite and compacts job
journals. It keeps the JSON cache as a backup. After you inspect the migrated
jobs, the second command verifies the entries in SQLite and deletes the legacy
JSON files. Both commands are safe to rerun.

`embed cache clean` removes the packed cache. A later resume re-embeds any
successful row whose journal key no longer resolves, which may incur provider
costs.

`--limit` is supported on `plan` and `run` for cheap local development.

`embed cache gc` is not in V1.

## License

EmbedForge is licensed under the MIT License. Datasets produced with EmbedForge are not automatically covered by the MIT License. Each generated dataset retains the license, attribution requirements, and usage restrictions of its source dataset, along with any applicable terms associated with the embedding model or service used to generate the embeddings.
