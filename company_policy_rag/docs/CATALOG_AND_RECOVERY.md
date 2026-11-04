# SQLite catalog and ingestion recovery — implementation checkpoint

Branch: `codex/production-critical-corrections`. Docker is excluded.

The new standard-library SQLite catalog stores document IDs, original paths and
hashes, options, expected chunk counts, jobs, leases, attempts and lifecycle events
separately from telemetry. It is integrated with API ingestion, retry, deletion,
startup restoration and retrieval visibility. Recovery replays the preserved
source after validation; legacy indexes are imported without automatic reindex.

This checkpoint is work in progress. Catalog/process recovery checks passed 19
tests, including actual abrupt process death. The expanded core suite passed 623
tests with one optional training-dependency skip. Existing queue-failure tests
now check source/job retention when an executor is unavailable. Selected Python
lint passed. Further rollback, cache/asset visibility and heartbeat regressions
have been added, and the final combined run is in progress.

Two old API tests assumed synchronous completion; they now poll the established
asynchronous upload/status contract. Remaining verification: their rerun, the
new final edge-case regressions, combined checks and operator recovery runbook.

No existing user library was migrated or reindexed during implementation.
Pre-existing user edits remain separate. The earlier architectural proposal's
catalog/recovery section is now being implemented; its other sections remain
proposals.
