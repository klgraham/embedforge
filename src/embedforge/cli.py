"""Composable EmbedForge CLI. No mega-command."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from embedforge.cache import EmbeddingCache
from embedforge.catalog import find_models, list_models
from embedforge.config import (
    apply_overrides,
    format_secret_status,
    get_value,
    load_config,
    reset_config,
    secret_env_status,
    set_value,
    unset_value,
)
from embedforge.errors import EmbedForgeError
from embedforge.inspect import inspect_dataset
from embedforge.paths import cache_dir, config_path
from embedforge.plan import build_plan
from embedforge.publish import publish_job
from embedforge.run import resume_job, start_or_resume
from embedforge.shapes import (
    InspectReport,
    Job,
    Plan,
    PublishResult,
    ValidationResult,
)
from embedforge.store import JobStore
from embedforge.validate import validate_job


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return int(args.handler(args))
    except EmbedForgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="embed",
        description=("Inspect, plan, embed, validate, and publish Hugging Face datasets."),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_set = sub.add_parser("set", help="set a non-secret config value")
    p_set.add_argument("key")
    p_set.add_argument("value")
    p_set.set_defaults(handler=cmd_set)

    p_get = sub.add_parser("get", help="print a config value")
    p_get.add_argument("key")
    p_get.set_defaults(handler=cmd_get)

    p_unset = sub.add_parser("unset", help="remove a stored config value")
    p_unset.add_argument("key")
    p_unset.set_defaults(handler=cmd_unset)

    p_reset = sub.add_parser("reset", help="clear all stored config")
    p_reset.set_defaults(handler=cmd_reset)

    p_config = sub.add_parser("config", help="show settings and secret availability")
    p_config.add_argument("--json", action="store_true")
    p_config.set_defaults(handler=cmd_config)

    p_inspect = sub.add_parser("inspect", help="inspect a source dataset")
    p_inspect.add_argument("dataset")
    p_inspect.add_argument("--split")
    p_inspect.add_argument("--config")
    p_inspect.add_argument("--json", action="store_true")
    p_inspect.set_defaults(handler=cmd_inspect)

    p_models = sub.add_parser("models", help="list known embedding models")
    p_models.add_argument("--provider")
    p_models.set_defaults(handler=cmd_models, models_cmd=None)
    models_sub = p_models.add_subparsers(dest="models_cmd")
    p_models_info = models_sub.add_parser("info", help="show one model")
    p_models_info.add_argument("model")
    p_models_info.add_argument("--provider")
    p_models_info.set_defaults(handler=cmd_models_info)

    p_plan = sub.add_parser("plan", help="show the job without embedding")
    _add_job_flags(p_plan, require_dataset=True)
    p_plan.add_argument("--json", action="store_true")
    p_plan.set_defaults(handler=cmd_plan)

    p_run = sub.add_parser("run", help="embed locally; do not publish")
    _add_job_flags(p_run, require_dataset=False)
    p_run.add_argument("--resume")
    p_run.set_defaults(handler=cmd_run)

    p_status = sub.add_parser("status", help="list jobs or show one job")
    p_status.add_argument("job_id", nargs="?")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(handler=cmd_status)

    p_resume = sub.add_parser("resume", help="continue a local job")
    p_resume.add_argument("job_id")
    p_resume.set_defaults(handler=cmd_resume)

    p_validate = sub.add_parser("validate", help="check a staged job")
    p_validate.add_argument("job_id")
    p_validate.add_argument("--json", action="store_true")
    p_validate.set_defaults(handler=cmd_validate)

    p_publish = sub.add_parser("publish", help="push a staged job (private by default)")
    p_publish.add_argument("job_id")
    p_publish.add_argument("--repo")
    p_publish.add_argument("--private", action="store_true")
    p_publish.add_argument("--public", action="store_true")
    p_publish.add_argument("--revision")
    p_publish.set_defaults(handler=cmd_publish)

    p_cache = sub.add_parser("cache", help="inspect the embedding cache")
    p_cache.set_defaults(handler=cmd_cache_help, cache_cmd=None)
    cache_sub = p_cache.add_subparsers(dest="cache_cmd")
    cache_info = cache_sub.add_parser("info", help="cache size and path")
    cache_info.set_defaults(handler=cmd_cache_info)
    cache_list = cache_sub.add_parser("list", help="list cache keys")
    cache_list.set_defaults(handler=cmd_cache_list)
    cache_clean = cache_sub.add_parser("clean", help="delete cached embeddings")
    cache_clean.set_defaults(handler=cmd_cache_clean)

    return parser.parse_args(list(argv))


def _add_job_flags(parser: argparse.ArgumentParser, *, require_dataset: bool) -> None:
    parser.add_argument("dataset", nargs=None if require_dataset else "?")
    parser.add_argument("--column")
    parser.add_argument("--output-column")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--config")
    parser.add_argument("--split")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--limit", type=int)


def cmd_set(args: argparse.Namespace) -> int:
    set_value(args.key, args.value)
    print(f"{args.key} = {get_value(args.key)}")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    print(get_value(args.key))
    return 0


def cmd_unset(args: argparse.Namespace) -> int:
    unset_value(args.key)
    print(f"unset {args.key}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    reset_config()
    print("config reset")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    config = load_config()
    secrets = secret_env_status()
    payload = {
        **config.to_dict(),
        "secrets": secrets,
        "config_path": str(config_path()),
        "cache_path": str(cache_dir()),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"Provider:        {config.provider}")
    print(f"Model:           {config.model}")
    print(f"Batch size:      {config.batch_size}")
    print(f"Concurrency:     {config.concurrency}")
    print(f"Dimensions:      {config.dimensions if config.dimensions else 'model default'}")
    print(f"Output column:   {config.output_column}")
    print(f"HF namespace:    {config.hf_namespace or 'not set'}")
    print()
    print(f"OPENAI_API_KEY:  {format_secret_status(secrets['OPENAI_API_KEY'])}")
    print(f"OPENROUTER_API_KEY: {format_secret_status(secrets['OPENROUTER_API_KEY'])}")
    print(f"HF_TOKEN:        {format_secret_status(secrets['HF_TOKEN'])}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    report = inspect_dataset(args.dataset, config=args.config, split=args.split)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0
    print(_format_inspect(report))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    if args.models_cmd == "info":
        return cmd_models_info(args)
    rows = list_models(args.provider)
    print(f"{'PROVIDER':<12} {'MODEL':<36} {'DIMENSIONS':>10}")
    for item in rows:
        print(f"{item.provider.value:<12} {item.model:<36} {item.dimensions:>10}")
    return 0


def cmd_models_info(args: argparse.Namespace) -> int:
    matches = find_models(args.model, provider=getattr(args, "provider", None))
    if not matches:
        raise EmbedForgeError(f"unknown model {args.model!r}")
    for item in matches:
        configurable = (
            f"yes ({item.min_dimensions}–{item.dimensions})"
            if item.configurable_dimensions
            else "no"
        )
        print(f"Model:                   {item.model}")
        print(f"Provider:                {item.provider.value}")
        print(f"Dimensions:              {item.dimensions}")
        print(f"Configurable dimensions: {configurable}")
        print(f"Max input length:        {item.max_input_tokens} tokens")
        print()
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    if not args.column:
        raise EmbedForgeError("--column is required")
    config = apply_overrides(
        load_config(),
        provider=args.provider,
        model=args.model,
        dimensions=args.dimensions,
        output_column=args.output_column,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )
    plan = build_plan(
        args.dataset,
        column=args.column,
        config=config,
        provider=config.provider,
        model=config.model,
        dimensions=config.dimensions,
        dataset_config=args.config,
        split=args.split,
        output_column=config.output_column,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return 0
    print(_format_plan(plan))
    print()
    print("No embedding API calls were made.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.resume and (not args.dataset or not args.column):
        raise EmbedForgeError("dataset and --column are required unless --resume is set")
    config = apply_overrides(
        load_config(),
        provider=args.provider,
        model=args.model,
        dimensions=args.dimensions,
        output_column=args.output_column,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )
    result = start_or_resume(
        args.dataset,
        column=args.column,
        resume_id=args.resume,
        config=config,
        provider=config.provider,
        model=config.model,
        dimensions=config.dimensions,
        dataset_config=args.config,
        split=args.split,
        output_column=config.output_column,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        limit=args.limit,
        on_progress=print,
    )
    print(f"job: {result.job.id}")
    print(f"status: {result.job.status.value}")
    print(f"output: {result.output_path}")
    return 0 if result.job.progress.failed == 0 else 1


def cmd_status(args: argparse.Namespace) -> int:
    store = JobStore()
    if args.job_id:
        job = store.load(args.job_id)
        payload = _job_status_dict(job, store)
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0
        print(_format_job(job, store))
        return 0
    jobs = store.list_jobs()
    if args.json:
        print(json.dumps([_job_status_dict(job, store) for job in jobs], indent=2))
        return 0
    if not jobs:
        print("no jobs")
        return 0
    print(f"{'JOB':<28} {'STATUS':<12} {'DATASET':<28} {'PROGRESS'}")
    for job in jobs:
        progress = f"{job.progress.next_index} rows"
        print(f"{job.id:<28} {job.status.value:<12} {job.source.repository:<28} {progress}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    result = resume_job(args.job_id, on_progress=print)
    print(f"job: {result.job.id}")
    print(f"status: {result.job.status.value}")
    print(f"output: {result.output_path}")
    return 0 if result.job.progress.failed == 0 else 1


def cmd_validate(args: argparse.Namespace) -> int:
    result = validate_job(args.job_id)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ok else 1
    print(_format_validation(result))
    return 0 if result.ok else 1


def cmd_publish(args: argparse.Namespace) -> int:
    if args.private and args.public:
        raise EmbedForgeError("choose --private or --public, not both")
    private = True
    if args.public:
        private = False
    if args.private:
        private = True
    result = publish_job(
        args.job_id,
        repo=args.repo,
        private=private,
        allow_public=bool(args.public),
        revision=args.revision,
    )
    print(_format_publish(result))
    return 0


def cmd_cache_help(args: argparse.Namespace) -> int:
    raise EmbedForgeError("usage: embed cache {info,list,clean}")


def cmd_cache_info(args: argparse.Namespace) -> int:
    info = EmbeddingCache().info()
    print(f"Path:    {info['path']}")
    print(f"Entries: {info['entries']}")
    print(f"Size:    {_format_bytes(int(info['bytes']))}")
    return 0


def cmd_cache_list(args: argparse.Namespace) -> int:
    entries = EmbeddingCache().list_entries()
    if not entries:
        print("cache is empty")
        return 0
    print(f"{'KEY':<20} {'PROVIDER':<12} {'MODEL':<32} {'DIMS'}")
    for entry in entries:
        print(f"{entry.key[:16]:<20} {entry.provider:<12} {entry.model:<32} {entry.dimensions}")
    return 0


def cmd_cache_clean(args: argparse.Namespace) -> int:
    removed = EmbeddingCache().clean()
    print(f"removed {removed} cached embeddings")
    print("note: embed cache gc is planned for a later release")
    return 0


def _format_inspect(report: InspectReport) -> str:
    splits = ", ".join(f"{name} ({count})" for name, count in report.splits.items()) or "none"
    columns = ", ".join(f"{col.name} ({col.dtype})" for col in report.columns) or "none"
    candidates = ", ".join(report.candidate_text_columns) or "none"
    return "\n".join(
        [
            f"Repository:   {report.repository}",
            f"Revision:     {report.revision}",
            f"License:      {report.license or 'unknown'}",
            f"Config:       {report.config or 'default'}",
            f"Split:        {report.split or 'unknown'}",
            f"Configs:      {', '.join(report.configs) or 'none'}",
            f"Splits:       {splits}",
            f"Columns:      {columns}",
            f"Text columns: {candidates}",
            "Estimated embedding input:",
            f"  rows:         {report.estimated_rows}",
            f"  characters:   {report.estimated_characters}",
        ]
    )


def _format_plan(plan: Plan) -> str:
    source = plan.source
    embedding = plan.embedding
    estimates = plan.estimates
    return "\n".join(
        [
            "Source",
            f"  Dataset:     {source.repository}",
            f"  Revision:    {source.revision}",
            f"  Config:      {source.config or 'default'}",
            f"  Split:       {source.split or 'all'}",
            f"  Column:      {', '.join(embedding.source_columns)}",
            "",
            "Embedding",
            f"  Provider:    {embedding.provider}",
            f"  Model:       {embedding.model}",
            f"  Dimensions:  {embedding.dimensions}",
            f"  Batch size:  {embedding.batch_size}",
            f"  Concurrency: {embedding.concurrency}",
            "",
            "Output",
            f"  Repo:        {plan.output.repo}",
            f"  Column:      {plan.output.column}",
            f"  Shape:       {estimates.rows} rows × {embedding.dimensions} dims",
            "",
            "Estimates",
            f"  Rows:        {estimates.rows}",
            f"  Characters:  {estimates.characters}",
            f"  Tokens:      {estimates.tokens}  (~4 chars/token)",
            f"  Cost:        ${estimates.cost_usd:.6f}",
        ]
    )


def _job_status_dict(job: Job, store: JobStore) -> dict[str, Any]:
    return {
        **job.to_dict(),
        "path": str(store.directory(job.id)),
        "output": str(store.output_dir(job.id)),
    }


def _format_job(job: Job, store: JobStore) -> str:
    progress = job.progress
    return "\n".join(
        [
            f"Job:          {job.id}",
            f"Status:       {job.status.value}",
            f"Dataset:      {job.source.repository}",
            f"Revision:     {job.source.revision}",
            f"Model:        {job.embedding.model}",
            f"Provider:     {job.embedding.provider}",
            f"Progress:     next={progress.next_index}",
            f"Embedded:     {progress.embedded}",
            f"Skipped:      {progress.skipped}",
            f"Failed:       {progress.failed}",
            f"Tokens:       {progress.tokens}",
            f"Cost:         ${progress.cost_usd:.6f}",
            f"Path:         {store.directory(job.id)}",
            f"Output:       {store.output_dir(job.id)}",
            f"Published:    {job.published_repo or 'not published'}",
            f"Error:        {job.error or 'none'}",
        ]
    )


def _format_validation(result: ValidationResult) -> str:
    lines = ["Checks"]
    for check in result.checks:
        mark = "ok" if check.passed else "FAIL"
        lines.append(f"  [{mark}] {check.name}: {check.message}")
    lines.append("")
    lines.append("Diagnostics")
    for item in result.diagnostics:
        lines.append(f"  {item.name}: {item.message}")
    lines.append("")
    lines.append("valid" if result.ok else "invalid")
    return "\n".join(lines)


def _format_publish(result: PublishResult) -> str:
    visibility = "private" if result.private else "public"
    lines = [
        f"Repo:       {result.repo_id}",
        f"Visibility: {visibility}",
        f"URL:        {result.url}",
    ]
    if result.revision:
        lines.append(f"Revision:   {result.revision}")
    return "\n".join(lines)


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"
