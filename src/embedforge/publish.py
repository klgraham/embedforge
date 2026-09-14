"""Publish a staged job to the Hugging Face Hub. Default visibility is private."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from datasets import DatasetDict, load_from_disk
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, hf_hub_download, whoami

from embedforge import __version__
from embedforge.errors import EmbedForgeError
from embedforge.hfdata import DatasetSource, hf_token
from embedforge.plan import default_output_repo
from embedforge.shapes import Job, JobStatus, PublishResult, replace_job
from embedforge.store import JobStore
from embedforge.validate import validate_job


@dataclass(frozen=True)
class DestinationInfo:
    exists: bool
    private: bool | None


class HubClient(Protocol):
    def inspect_destination(self, repo_id: str, token: str | None) -> DestinationInfo: ...

    def push_dataset(
        self,
        dataset_dir: Path,
        repo_id: str,
        *,
        private: bool,
        revision: str | None,
        config_name: str | None,
        token: str | None,
    ) -> None: ...

    def read_text(
        self,
        path_in_repo: str,
        repo_id: str,
        *,
        revision: str | None,
        token: str | None,
    ) -> str: ...

    def upload_text(
        self,
        content: str,
        path_in_repo: str,
        repo_id: str,
        *,
        revision: str | None,
        token: str | None,
    ) -> None: ...


class DestinationLookup(Protocol):
    def repo_exists(
        self, repo_id: str, *, repo_type: str | None = None, token: str | None = None
    ) -> bool: ...

    def dataset_info(self, repo_id: str, token: str | None = None) -> Any: ...


class HuggingFaceHub:
    def __init__(self, api: DestinationLookup | None = None) -> None:
        self._api = api

    def _client(self, token: str | None) -> DestinationLookup:
        return self._api if self._api is not None else HfApi(token=token)

    def inspect_destination(self, repo_id: str, token: str | None) -> DestinationInfo:
        api = self._client(token)
        try:
            exists = api.repo_exists(repo_id, repo_type="dataset", token=token)
        except Exception as exc:
            raise EmbedForgeError(f"could not inspect destination {repo_id}: {exc}") from exc
        if not exists:
            return DestinationInfo(exists=False, private=None)
        try:
            info = api.dataset_info(repo_id, token=token)
        except Exception as exc:
            raise EmbedForgeError(f"could not inspect destination {repo_id}: {exc}") from exc
        return DestinationInfo(exists=True, private=bool(getattr(info, "private", False)))

    def push_dataset(
        self,
        dataset_dir: Path,
        repo_id: str,
        *,
        private: bool,
        revision: str | None,
        config_name: str | None,
        token: str | None,
    ) -> None:
        if not token:
            raise EmbedForgeError(
                "API credentials must be supplied through environment variables.\n"
                "Set HF_TOKEN in your shell environment."
            )
        dataset = load_from_disk(str(dataset_dir))
        resolved_config = config_name or "default"
        if revision is None:
            dataset.push_to_hub(repo_id, private=private, token=token, config_name=resolved_config)
        else:
            dataset.push_to_hub(
                repo_id,
                private=private,
                token=token,
                revision=revision,
                config_name=resolved_config,
            )

    def upload_text(
        self,
        content: str,
        path_in_repo: str,
        repo_id: str,
        *,
        revision: str | None,
        token: str | None,
    ) -> None:
        if not token:
            raise EmbedForgeError(
                "API credentials must be supplied through environment variables.\n"
                "Set HF_TOKEN in your shell environment."
            )
        api = HfApi(token=token)
        if revision is None:
            api.upload_file(
                path_or_fileobj=content.encode("utf-8"),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
        else:
            api.upload_file(
                path_or_fileobj=content.encode("utf-8"),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                revision=revision,
            )

    def read_text(
        self,
        path_in_repo: str,
        repo_id: str,
        *,
        revision: str | None,
        token: str | None,
    ) -> str:
        if not token:
            raise EmbedForgeError(
                "API credentials must be supplied through environment variables.\n"
                "Set HF_TOKEN in your shell environment."
            )
        try:
            downloaded = hf_hub_download(
                repo_id,
                path_in_repo,
                repo_type="dataset",
                revision=revision,
                token=token,
            )
        except Exception as exc:
            raise EmbedForgeError(f"could not read {path_in_repo} from {repo_id}: {exc}") from exc
        return Path(downloaded).read_text(encoding="utf-8")


def merge_dataset_card(generated: str, ours: str) -> str:
    """Keep Hub configs/dataset_info and overlay EmbedForge license plus body."""
    generated_card = DatasetCard(generated, ignore_metadata_errors=True)
    our_card = DatasetCard(ours, ignore_metadata_errors=True)
    merged = generated_card.data.to_dict()
    license_value = our_card.data.to_dict().get("license")
    if license_value is not None:
        merged["license"] = license_value
    body = our_card.text.lstrip("\n")
    return f"---\n{DatasetCardData(**merged).to_yaml()}\n---\n{body}"


@dataclass
class HuggingFacePublisher:
    hub: HubClient = field(default_factory=HuggingFaceHub)

    def publish(
        self,
        *,
        repo_id: str,
        private: bool,
        allow_public: bool,
        revision: str | None,
        dataset_dir: Path,
        card: str,
        provenance_yaml: str,
        config_name: str | None,
        token: str | None,
    ) -> tuple[str, bool]:
        destination = self.hub.inspect_destination(repo_id, token)
        if destination.exists and destination.private is False and not allow_public:
            raise EmbedForgeError(
                f"destination {repo_id} is public. Pass --public to publish there."
            )
        self.hub.push_dataset(
            dataset_dir,
            repo_id,
            private=private,
            revision=revision,
            config_name=config_name,
            token=token,
        )
        generated = self.hub.read_text(
            "README.md",
            repo_id,
            revision=revision,
            token=token,
        )
        merged_card = merge_dataset_card(generated, card)
        self.hub.upload_text(
            merged_card,
            "README.md",
            repo_id,
            revision=revision,
            token=token,
        )
        self.hub.upload_text(
            provenance_yaml,
            "embedforge.yaml",
            repo_id,
            revision=revision,
            token=token,
        )
        actual = self.hub.inspect_destination(repo_id, token)
        if actual.exists and actual.private is not None:
            reported_private = actual.private
        else:
            reported_private = private
        return f"https://huggingface.co/datasets/{repo_id}", reported_private


def resolve_namespace(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    token = hf_token()
    if not token:
        return None
    try:
        identity = whoami(token=token)
    except Exception:
        return None
    if isinstance(identity, dict):
        name = identity.get("name")
        return name if isinstance(name, str) else None
    name = getattr(identity, "name", None)
    return name if isinstance(name, str) else None


def build_dataset_card(job: Job, provenance_yaml: str) -> str:
    source = job.source
    embedding = job.embedding
    source_url = f"https://huggingface.co/datasets/{source.repository}"
    license_line = source.license or "unknown"
    split_line = source.split or "all"
    return (
        f"---\n"
        f"license: {license_line}\n"
        f"---\n\n"
        f"# {job.output.repo.split('/')[-1]}\n\n"
        f"Derived embedding dataset generated by "
        f"[EmbedForge](https://github.com/klgraham/embedforge) "
        f"{__version__}.\n\n"
        f"## Source\n\n"
        f"- Dataset: [{source.repository}]({source_url})\n"
        f"- Revision: `{source.revision}`\n"
        f"- Config: `{source.config or 'default'}`\n"
        f"- Split: `{split_line}`\n"
        f"- License: `{license_line}` (inherited from the source dataset)\n\n"
        f"## Embedding\n\n"
        f"- Provider: `{embedding.provider}`\n"
        f"- Model: `{embedding.model}`\n"
        f"- Dimensions: `{embedding.dimensions}`\n"
        f"- Source column(s): {', '.join(f'`{name}`' for name in embedding.source_columns)}\n"
        f"- Output column: `{embedding.column}`\n\n"
        f"## Provenance\n\n"
        f"```yaml\n{provenance_yaml.rstrip()}\n```\n\n"
        f"## License\n\n"
        f"This dataset retains the license, attribution requirements, and usage "
        f"restrictions of its source dataset, along with any applicable terms "
        f"associated with the embedding model or service used to generate the "
        f"embeddings.\n"
    )


def publish_job(
    job_id: str,
    *,
    repo: str | None = None,
    private: bool = True,
    allow_public: bool = False,
    revision: str | None = None,
    store: JobStore | None = None,
    hub: HubClient | None = None,
    source: DatasetSource | None = None,
    namespace: str | None = None,
) -> PublishResult:
    job_store = store or JobStore()
    job = job_store.load(job_id)
    output = job_store.output_dir(job_id)
    if not output.exists():
        raise EmbedForgeError(f"job {job_id} has no staged output; run it first")
    if job.status not in {JobStatus.COMPLETED, JobStatus.VALIDATED}:
        raise EmbedForgeError(
            f"job {job_id} is {job.status.value}; publish requires a completed job"
        )
    if job.progress.failed:
        raise EmbedForgeError(
            f"job {job_id} has {job.progress.failed} unresolved failures; resume before publishing"
        )
    validation = validate_job(job_id, store=job_store, source=source, mark_status=True)
    if not validation.ok:
        raise EmbedForgeError("job failed validation; not publishing")
    job = job_store.load(job_id)
    provenance = job_store.load_provenance(job_id)
    card = build_dataset_card(job, provenance.to_yaml())
    job_store.write_card(job_id, card)
    repo_id = repo or job.output.repo
    if "/" not in repo_id:
        resolved_ns = resolve_namespace(namespace)
        if resolved_ns:
            repo_id = f"{resolved_ns}/{repo_id}"
        else:
            repo_id = default_output_repo(
                job.source.repository,
                job.embedding.provider,
                job.embedding.model,
                namespace=resolved_ns,
            )
            if "/" not in repo_id:
                raise EmbedForgeError("pass --repo owner/name or `embed set hf_namespace NAME`")
    _assert_staged_structure(output)
    publisher = HuggingFacePublisher(hub=hub or HuggingFaceHub())
    url, actual_private = publisher.publish(
        repo_id=repo_id,
        private=private,
        allow_public=allow_public,
        revision=revision,
        dataset_dir=output,
        card=card,
        provenance_yaml=provenance.to_yaml(),
        config_name=job.source.config,
        token=hf_token(),
    )
    job_store.save_job(replace_job(job, status=JobStatus.PUBLISHED, published_repo=repo_id))
    return PublishResult(
        repo_id=repo_id,
        private=actual_private,
        revision=revision,
        url=url,
        card=card,
    )


def _assert_staged_structure(output: Path) -> None:
    staged = load_from_disk(str(output))
    if not isinstance(staged, DatasetDict) or not list(staged.keys()):
        raise EmbedForgeError("staged output is missing split names")
