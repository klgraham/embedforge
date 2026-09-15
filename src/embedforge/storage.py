"""One-time migration from decimal JSON embeddings to packed storage."""

from __future__ import annotations

from dataclasses import dataclass

from embedforge.cache import CacheMigration, EmbeddingCache
from embedforge.store import JobStore, JournalCompaction


@dataclass(frozen=True)
class StorageMigration:
    cache: CacheMigration
    journals: tuple[tuple[str, JournalCompaction], ...]

    @property
    def bytes_before(self) -> int:
        return sum(result.bytes_before for _job_id, result in self.journals)

    @property
    def bytes_after(self) -> int:
        return sum(result.bytes_after for _job_id, result in self.journals)


def migrate_storage(
    *,
    cache: EmbeddingCache | None = None,
    store: JobStore | None = None,
    delete_legacy_cache: bool = False,
) -> StorageMigration:
    embedding_cache = cache or EmbeddingCache()
    job_store = store or JobStore()
    cache_result = embedding_cache.migrate_legacy(delete=delete_legacy_cache)
    journals = tuple(
        (job.id, job_store.compact_journal(job.id, embedding_cache))
        for job in job_store.list_jobs()
    )
    return StorageMigration(cache=cache_result, journals=journals)
