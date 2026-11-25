# Authoring v2: larger experiment budgets and selective evidence

This is the new version of the provider-neutral `authoring` entry. Old `authoring`
(public skill knowledge v1), baseline and Option-search-v1/v2/v3 are frozen historical
entries. Their files/seals are unchanged. All new backend comparisons should use the
same authoring_v2 information and budget profile; task/control/search behavior is not
changed. Read [experiment_api.md](experiment_api.md) for the full tool contract.

For the guarded planar transport research line, author the option effects and
successor obligations using the [symbolic transport protocol](symbolic_transport.md).

```python
from experiments.planar_tasks.authoring_v2 import campaign_config, create_session, drive, verify_frozen
config = campaign_config(explicit_model_dict, development_tasks=three_or_nine_allowed_names)
s = create_session(new_directory, config)  # method='option_search_v3', or 'free'
try:
    result = drive(s, explicitly_selected_transport)
finally:
    s.close()
verify_frozen(new_directory)  # before calling the original evaluator
```

Creation does not step physics or call a provider. Run hosts as detached supervised
processes from the outset (preserving venv executable path rather than resolving its
symlink), not as turn-owned long-running exec cells. This release does not implement
automatic resume or pause-time refunds. Never interpret a stale running JSON as proof
of a live host. Infrastructure interruption must remain explicit evidence.

Default proposed 90-minute configuration: 2000 explicit resets, 100 policy tests,
250000 executed ticks, 4200 script seconds, 80 model requests, 32 MiB cumulative model
input bytes, 1310720 reserved output tokens at up to 16384/request. Final submit-only
reserve is 300 seconds/2 requests/32768 output tokens/1 MiB input/12 tools, inside the
total. Budgets are configurable before a session; provider token needs may justify
explicitly different declared caps. No new experiment is launched by this change.

Development prediction totals increase to 1000000 ticks/2400 CPU seconds, while
per-decision, per-episode and deployment planner limits stay frozen. Full-test reset
counts are separate from explicit resets. Known test/script completion settles actual
work; missing trustworthy results retains reservations. Initial quota admission stays
conservative. Failed/partial work is counted, never made free by returning early.

Batch experiments replay prefixes separately for each branch; every tick is charged.
There is no privileged live-state clone, independent runtime execution oracle, hidden
truth summary, task-specific success script or reward adjustment. Summaries are the
same for all LLM backends and free/search authoring. Scripts can define additional
metrics from legal data themselves.

The deterministic context index is lossy and labeled. Raw public tool records and
original model events stay durable; the author notebook is explicitly unverified.
Native relay context cannot be forcibly compacted by this host. API byte reductions
are measurable offline; token/billing improvements require a real metered run.

Freeze/verify via `release.freeze()` / `release.verify()`. Submit manifests reference
both the original knowledge profile and the new experiment profile. Use v2 verifiers
for v2 sessions. No old result is reinterpreted under these new accounting rules.
