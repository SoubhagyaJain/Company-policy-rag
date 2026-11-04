# SQLite catalog and ingestion recovery — implementation checkpoint

Branch: `codex/production-critical-corrections`. Docker is excluded.

The new standard-library SQLite catalog stores document IDs, original paths and
hashes, options, expected chunk counts, jobs, leases, attempts and lifecycle events
separately from telemetry. It is integrated with API ingestion, retry, deletion,
startup restoration and retrieval visibility. Recovery replays the preserved
source after validation; legacy indexes are imported without automatic reindex.

This checkpoint is work in progress. Initial existing regressions: 48 passed,
one old queue-failure expectation needs updating for the deliberate durable-queue
behavior (retain the source/job when an executor is unavailable). Selected Python
lint passed before the new catalog tests were added.

Remaining verification: catalog transaction/lease tests; stage interruption and
restart tests; actual process termination; migration/identity preservation;
source/hash mismatch, partial cleanup, retry options and bounded attempts;
combined regression suite. Do not describe this checkpoint as validated release.

No existing user library was migrated or reindexed during implementation.
Pre-existing user edits remain separate. The earlier architectural proposal's
catalog/recovery section is now being implemented; its other sections remain
proposals.
