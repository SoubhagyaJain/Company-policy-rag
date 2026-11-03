# Critical corrections: validation and deployment

The application keeps the existing `app/storage` library location, document IDs,
and indexes. Compose now mounts that exact directory. No migration or reindex is
performed automatically. The separate `storage` mount is retained.

## Existing installations: preserve before deploying

Do not recreate an old container before preserving its writable-layer library.
Stop requests and wait for ingestion to finish. Stop (do not remove) the old
`rag-backend` container, and use `docker cp rag-backend:/app/app/storage <backup>`
to copy its library to a **new, empty backup directory**. Preserve the old Compose
file, container, and all named volumes. Inventory files and compare SHA-256 hashes
between the copy and original before proceeding.

If the host's `app/storage` is empty, restore the verified backup there. If it
already contains data, compare document IDs and hashes and reconcile into a
separate staging directory; never overwrite or merge conflicting libraries
automatically. Test the preserved copy first. Deploy only after checking source
bytes, document IDs, Chroma counts, BM25 retrieval, and citations. Keep the old
container and backup until the new deployment has been verified.

Rollback uses the preserved old Compose configuration and stopped container, or
the verified backup. Do not use `docker compose down -v` during this process.

## Persistence checks

Offline: `python -m pytest tests/unit/test_deployment_persistence.py tests/unit/test_clean_startup.py -q`.

Explicit Docker integration (uses only a disposable test directory):

```powershell
docker build -f docker/Dockerfile.backend -t policy-rag-persistence-test .
$env:RAG_CONTAINER_TEST_IMAGE = 'policy-rag-persistence-test'
python -m pytest tests/integration/test_container_library_recreation.py -q
```

The integration test creates two separate containers with networking disabled.
It ingests using deterministic test embeddings in the first and checks original
bytes, document IDs, vector rows, and restored BM25 chunks in the second.
