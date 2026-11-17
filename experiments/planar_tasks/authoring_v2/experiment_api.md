# Experiment authoring v2 — summary first, evidence on demand

Read this alongside the SAME option_guide.md and controller/search contracts. No new
physical skill, reward, execution observation or deployment search budget is introduced.
The host experiment_profile.json specifies this development-only information interface.

## Budgets and clock

The suggested campaign constructor gives 90 minutes total including a 5-minute
submit-only reserve, 2000 explicit development resets, 100 complete-policy test calls,
250000 execution ticks, 120 seconds per script and 4200 total script wall seconds.
Actual campaign.json is authoritative; model/provider/settings and caps are explicit.
Resets are NOT automatically cheap/free physics: every executed prefix and suffix tick
counts. test_policy resets have their own diagnostic counter, not the explicit-reset
quota. Each test reserves max_ticks for admission and settles to actual ticks when a
trustworthy summary exists; otherwise the full reservation remains. Scripts reserve
their requested allowed timeout and settle to measured wall time when a result exists.
Model output remains conservatively reserved, not refunded or claimed as actual tokens.

Shared development prediction totals (scripts + tested policy prediction): 100000 calls,
1000000 ticks, 30000 model creations, 2400 CPU seconds. This supersedes only the older
prediction_api.md DEVELOPMENT TOTALS. Live handles, per-act limits and each tested
policy episode's prediction limits stay unchanged, as do deployment search limits.
Budget reports provide current remaining counters. A declaration of a larger quota
never itself launches paid calls, physics, or a second campaign. No unlimited loop.

## Batch counterfactual experiments

`run_branches(task, prefix, candidates)` executes each candidate from a new reset of
the same allowed fixture after replaying the SAME command prefix. Commands have exact
integer a0/a1/ticks fields. prefix <=200 commands, each candidate 1..12 commands,
1..128 candidates per batch, <=120 real seconds per batch (also limited by remaining
development/script time). All replay ticks are charged. This is replay, not a hidden
snapshot API or arbitrary pose setter. Legality is checked as each command executes.
A declared batch must fit its worst-case remaining execution/reset quotas; split it
if necessary. Terminal prefixes, invalid actions and timeouts are retained as failures;
not-started branches are reported. Failed branches are never silently retried.

Default reply: attempted/completed/error counts, ticks, record ID and five-row preview.
Use `batch_rows(batch_id,start,count,sort_by,descending)` to inspect at most 20 rows.
Sort explicitly by branch, executed_ticks, estimated_translation or estimated_rotation_abs.
Sorting is descriptive, not a reward or a host-selected best policy. Missing estimates
sort last. A nonzero translation/rotation alone does not establish task value.

Within development scripts the same operation is available as:

```python
import sys
sys.path.insert(0, "/work")  # needed for scripts in experiments/ subdirectories
from batch_client import BatchEnv
# Choose legal commands from observations; these variable names are placeholders.
r = BatchEnv().branches(task_name, prefix_commands, candidate_command_lists)
print(r)  # already a compact overview, full observations are saved by the host
```

Ordinary DevelopmentEnv reset/step/observe remains available. Python scripts may
compute their own statistics and print concise tables instead of dumping sensors.
Only the existing policy/heuristic deployment files may be submitted; development
clients and experiments do not become a deployment simulator route.

## Three evidence levels

1. `test_summary(test_id)` returns public outcome, boundary-sampled stage/contact
   counts, estimated object change, robot net movement, missing-estimate count and
   fixed-search diagnostics when logged. It does NOT read truth.json, environment
   dumps, hidden maps, evaluator scores or full controller state.
2. `test_events(record_id, branch?, start=0, count=20)` indexes visible stage, contact,
   done, pause and direction-availability changes with row/time. A batch record needs
   a branch index. These are events BETWEEN OBSERVATION SAMPLES, not exact internal
   transition times or durations. Unseen transitions are not fabricated. Search tests also
   append prediction_feedback entries by decision ID, explicitly labeled as reported
   comparisons between predictions and subsequent legal estimates; noncomparable
   action durations do not receive a residual score.
3. `read_trace(record_id, branch?, start=0, count=5, fields?, robots?)` reads selected
   rows. fields: own, execution, contacts, scan, patches, goal, legal_pairs; default
   own/execution/contacts. robots defaults [0,1]. No arbitrary JSON paths. Responses
   page by row and byte cap; ask for fewer fields/robots if one row is too large.

For a batch trace row 0 is reset, followed by prefix and branch observations;
branch_start_row identifies the branch origin. For a policy test the sensor log contains
pre-decision observations: the last sample may precede the terminal action. End times
and sample counts make this visible. Contact sample counts are NOT contact seconds.
Estimated object deltas come from the frozen legal-observation rectangle estimator,
not actual object truth. Estimation can be missing, carried forward from an earlier sample, or wrong; valid
estimate time bounds and carried-estimate counts are reported. No true obstacle clearance
or hidden contact mode is exposed. Public terminal success remains the existing allowed
outcome from the test harness. `read_log` remains available for original legal logs.

## Durable results and context

Every tool request/result, including failures, receives an integer record_id and is
saved before returning. Replies larger than 12000 bytes are replaced by an explicit
archive notice. `list_results(start,count)` pages IDs/names/status; `read_result(record_id,
start,count)` reads up to 8000 bytes of the original PUBLIC tool exchange. This is not
an arbitrary filesystem reader. Returned next_start stays on a UTF-8 boundary; an arbitrary start inside a character
is rejected. Source hashes/raw logs remain authoritative. Workspace code can also be reread via read_file.

`notebook(text)` replaces up to 8000 bytes of author notes. Keep hypotheses, key evidence
IDs, unsuccessful checks, selected candidate identity and next questions here. These
are AUTHOR claims, not host-certified findings.

The automatic API driver uses a marked deterministic context projection once message
history exceeds 64000 bytes. It retains fixed instructions, current budgets, notebook,
recent complete tool exchanges where they fit, and an archive notice indexing older
outcomes/errors. Older detailed results AND old code-writing exchanges remain in the
original event/result records, rather than being sent repeatedly. Up to 131072 message
bytes are retained; metadata/schema overhead is additionally metered. Projection hashes
and byte sizes are recorded; no extra summarizer LLM is used. Failures are not deleted
from disk or silently replaced with a success narrative. Look up relevant evidence.

This projection controls provider API request bodies. A native external relay's own
model context is not managed by this host: incremental relay delivery avoids duplicate
old outputs, but internal context reclamation is NOT guaranteed. Summary-first tools
reduce new output for both paths. No actual token-price saving is claimed without
provider usage evidence. Do not equate bytes, reserved tokens and native model tokens.

Finalization still allows ONLY atomic submit; do not postpone final regression or
necessary evidence reads until that stage. The host never chooses or fixes your policy.
