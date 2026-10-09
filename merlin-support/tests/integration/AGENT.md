# AGENT.md — merlin-support/tests/integration

Migrated historical Gemmini tests, not lightweight migration qualification. Collection requires
`GEMMINI_RUN_INTEGRATION=1`; tests may then execute installed simulators. Do not enable casually.
`GEMMINI_LEGACY_FIXTURE_ROOT` explicitly selects the historical fixture checkout (see README).
It never selects a runtime backend: implementation is the explicitly selected support provider.
Do not fabricate missing facts, alter archived manifests, or weaken existing test assertions.
