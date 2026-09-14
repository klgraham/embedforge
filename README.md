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
embed plan DATASET --column text
embed run DATASET --column text --limit 32
embed status
embed resume JOB_ID
embed validate JOB_ID
embed publish JOB_ID
```

`plan` is a first-class paid-operation gate. It shows the source revision, output
shape, destination, and estimated cost, and it never calls an embedding API.

`run` writes a local job under `~/.cache/embedforge/jobs/<id>/` and does not
publish. `publish` defaults to a **private** Hugging Face dataset.

## Configuration

```bash
embed set provider openai
embed set model text-embedding-3-small
embed set batch_size 128
embed set concurrency 8
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
```

Embedding cache keys are `H(provider, model, dimensions, input)` so retries do
not re-pay identical values. Jobs use ULID-style ids and are resumable with
`embed resume JOB_ID` or `embed run --resume JOB_ID`.

`--limit` is supported on `plan` and `run` for cheap local development.

`embed cache gc` is not in V1.

## License

EmbedForge is licensed under the MIT License. Datasets produced with EmbedForge are not automatically covered by the MIT License. Each generated dataset retains the license, attribution requirements, and usage restrictions of its source dataset, along with any applicable terms associated with the embedding model or service used to generate the embeddings.
