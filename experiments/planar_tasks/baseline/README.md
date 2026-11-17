# Autonomous baseline tools v1.3.6

The normative definition is [Agentic_Options_Baseline_v1.md](../../../docs/research/Agentic_Options_Baseline_v1.md).
This package is separate from the historical `agentic.py` fixed-feedback pilot.
It freezes **infrastructure**, not a final task benchmark or model performance.
No model API call or physical rollout is started on import, initialization, or release verification.

The agent decides which scripts/experiments to run and explicitly selects one final
submission. It may independently implement history, models, search and recovery.
No handcrafted planner, demonstration, experience library, failure classifier,
automatic action repair or best-candidate selector is supplied.

## Interfaces

`driver.TOOLS` exposes `list_files`, `read_file`, `write_file`, `run_python`,
`test_policy`, `read_log`, and `submit`. A workspace has an inspectable `client.py`:

```python
from client import DevelopmentEnv

env = DevelopmentEnv()
result = env.reset(task_name)   # agent chooses among configured development fixtures
obs = result["observation"]
result = env.step(action)      # exact {a0, a1, ticks}; no fallback for invalid actions
svg = env.render()             # raw visible scan points and public poses/goal only
```

The simulator lives outside the sandbox. The client has no source/state getter,
privileged reset, test panel, full-map renderer, or reward/hidden-distance endpoint.
Every reset, tool/RPC, actual step tick and test reservation is charged. Script
stdout/stderr and full RPC history are queryable by returned `run_id`; policy-test
sensors/decisions/stderr by `test_id`. Log reads are bounded and paginated.

Deployment authors implement:

```python
class Policy:
    def __init__(self, contract):
        ...  # Initialize this episode; no host environment access.

    def act(self, observation):
        ...  # Return exactly {"a0": int, "a1": int, "ticks": int}.
```

The frozen `policy_host.py` only handles JSONL, initialization and flushing. All
choices belong to the authored program. Submit `policy.py` and any self-authored
support modules/data via an explicit file list. submit also accepts an optional
contents map of listed filenames to complete text, atomically staging new/changed
files without a prior write_file call. Syntax, isolated construction and act signature
are checked before freezing; no act call or physics validation occurs. Tests archive copies before
execution; submission does not pick a test winner. Each deployment starts a fresh
process with a read-only package, fresh `/tmp`, and no development RPC or network.
`verify_frozen()` rejects altered files/config/release before evaluation.

## Initialization and tool service

```bash
.venv/bin/python -m experiments.planar_tasks.baseline.cli verify
.venv/bin/python -m experiments.planar_tasks.baseline.cli init \
  --config experiments/planar_tasks/baseline/smoke_config.json \
  --session outputs/agentic/my_new_tools_smoke
.venv/bin/python -m experiments.planar_tasks.baseline.cli serve \
  --session outputs/agentic/my_new_tools_smoke
```

`serve` accepts one JSON tool request per line, e.g.
`{"name":"list_files","arguments":{}}`. It is a long-lived **trusted host** tool
process, not a shell exposed to the generated code. Environment state persists
only while this host lives. Reopening preserves budgets and logs, but requires a
fresh reset; it does not restore a privileged simulator snapshot.

The supplied `infrastructure_smoke` config rejects real simulator reset/testing.
Unit tests inject synthetic worlds. A real campaign must explicitly provide a
`development` config, precise provider/model/settings, task list and all budgets.
All existing nine scenes remain development. New independent roots are generated
explicitly on the host before synthesis, never inside an agent tool. Changing purpose does not itself launch any work.

## Autonomous model orchestration

`driver.drive(session, complete)` is a provider-neutral tool-calling loop. The trusted
`complete(payload, timeout)` transport must return a Chat-Completions-shaped JSON
response. The model can choose any sequence of tools, including custom experiments;
there is no fixed run/feedback/revision schedule. Stop without explicit `submit`
means **no submission**, not automatic recovery of the best candidate.

A transport runs in a separately bounded host worker. Exact input messages, tools,
model responses/usage and tool results are logged outside the agent workspace.
Generated scripts never receive credentials or the transport. Returned model
identity must match the campaign model; differing aliases/versions require an
explicit campaign configuration, not silent substitution. The explicit `transports.deepseek_complete` adapter reuses only the existing
bounded HTTPS request transport, without the legacy fixed-feedback synthesis loop.
`cli run-deepseek --session ...` requires provider `deepseek` and purpose
`development`. `synthesize-deepseek` combines init + run without a human waiting gap.
`preflight-deepseek` makes one bounded billable native-tool check without physics.
Only these three explicit DeepSeek commands start model requests.
Credentials stay in the existing external 0600 credential file or host environment.
Native function calls and any provider reasoning continuation field are preserved.
Synthetic and mocked tests exercise the loop; live protocol status is recorded
separately in the infrastructure review, not inferred from those tests. See the [official tool protocol](https://api-docs.deepseek.com/guides/tool_calls/).
Other providers can implement the same trusted transport interface; exact adapter
source and model configuration must be frozen for any cross-provider campaign.

The strict information boundary applies to API tool-only authors and isolated
fresh coding-agent sessions. Running the tools from the present repository-informed
conversation does not erase prior source knowledge and must not be labelled blind.

## Budget and provenance semantics

All `Budget` fields are required, positive and finite; integer caps reject floats
and booleans. Hard campaign caps cover model calls, canonical input payload UTF-8
bytes, reserved maximum output tokens, tools/RPC, resets, simulator ticks, policy
tests, Python wall-time reservations, and total session wall time. A timed-out model
request retains its full reservation and unknown billing; there is no retry.
Provider-reported token usage is recorded separately. This is not exact monetary
accounting or a claim of equivalence to the paper's $20 budget. Dollar-matched
experiments require a separately frozen metered provider adapter/pricing policy.

Python and test calls reserve their full declared limits before execution. Unused
reservations are not refunded. Actual test ticks/time remain separately recorded.
This conservative rule prevents failures/restarts from avoiding resource accounting.
Model/token budgets are enforced by `drive`; external use of `serve` cannot meter
an outside Codex conversation and must be reported as external-author/tool-only.

Python scripts have a 512 MiB per-process address-space limit, per-process CPU
limit, 16 MiB file-size limit, bounded output, and whole-process-group termination.
Workspace size is checked on writes and after execution; it is **not an OS-level
aggregate disk quota**. `/tmp` is ephemeral. These restrictions provide information
isolation and finite experiment execution, not hostile-code aggregate resource
containment. Trusted RPC handlers check deadlines per physics tick; deadline
resolution is one simulator tick, and arbitrary blocking host handlers are not
supported. Failure of namespace isolation never triggers an unsandboxed fallback.

`release_v1_3_6.json` records source/content hashes, runtime binaries/dependency versions,
protocol and development catalog. `verify` rejects drift or added source files;
there is no auto-refresh command. New releases need a new version and lock, keeping
the old snapshot. This release includes a conservative dependency-source superset.
Program packages, configurations and a chained event log preserve provenance.
No commit is required to verify a dirty working tree's exact frozen contents.

The initial v1 release candidate is preserved in its snapshot. v1.1 adds provider
process-group cleanup so nested workers cannot survive a model timeout; its
descendant-process regression test passes. No model campaign ran on the candidate.


## Prepare and execute one campaign

The supplied `development_config.json` is an explicit, **not-yet-run** campaign:
deepseek-flash (current documented Flash-1017 alias), thinking disabled, 40 calls/30 minutes; caps are reservations, not
money or actual tokens. Its finalization reserve is included in those totals.
Tool results and each model prompt report remaining budgets. During finalization,
only submit is exposed and tool_choice forces submit; the agent still selects its
own files/content. Default: at most 38 development calls plus 2 submission calls.
Stage 2 has its own declared limits even when development ends early. A failed
syntax/import/interface check can be corrected within the remaining submission calls.
A reserve cannot guarantee submission after provider failures or invalid output.
External-author runs cannot claim API-token budget equivalence.

Prepare the panel **before** creating any synthesis session. All paths must be new:

```bash
.venv/bin/python -m experiments.planar_tasks.baseline.cli create-panel \
  --out outputs/agentic/panels/instance_transfer_v1.json --seed 2026110901 --per-family 10
# Explicit billable protocol check: one call, max 1024 output tokens, no physics.
.venv/bin/python -m experiments.planar_tasks.baseline.cli preflight-deepseek \
  --config experiments/planar_tasks/baseline/development_config.json \
  --out outputs/agentic/my_protocol_preflight
# Explicit billable autonomous synthesis; no automatic launch from the console.
.venv/bin/python -m experiments.planar_tasks.baseline.cli synthesize-deepseek \
  --config experiments/planar_tasks/baseline/development_config.json \
  --session outputs/agentic/my_campaign
# Only after the model explicitly submits; no LLM, no repair, no selection.
.venv/bin/python -m experiments.planar_tasks.baseline.cli evaluate \
  --session outputs/agentic/my_campaign \
  --panel outputs/agentic/panels/instance_transfer_v1.json \
  --budget experiments/planar_tasks/baseline/evaluation_config.json \
  --out outputs/agentic/my_evaluation
```

The 30-case panel samples fresh roots across three existing families. It measures
same-family instance transfer, not novel dynamics, shapes or task families. Initial
geometry checks do not establish solvability. Do not regenerate/select panels based
on outcomes. Evaluations count all scheduled instances, explicitly distinguish
attempted/errors/not-run, and reserve the submission/panel pair before any rollout;
interruption does not authorize a silent rerun. Programs start in fresh processes.

Read-only console (does not launch experiments or require port 8765):

```bash
.venv/bin/python -m experiments.planar_tasks.console.server --port 8766
```

Open http://127.0.0.1:8766/. It reuses the option testbed renderer for saved replays.


## v1.3 two-stage submission

Stage 1 uses the full tool menu. At its allowance boundary, stage 2 receives only
`submit` with forced function choice. This is an artifact-delivery protocol, not an
extra planner or recovery strategy. Its caps were reserved before the experiment;
no calls are added when an attempt fails. DeepSeek thinking mode cannot force a
specific function, so a DeepSeek finalization config must explicitly disable thinking.

```python
submit(files=["policy.py", "helper.py"], contents={
    "policy.py": "complete model-authored source here"
})
# helper.py is copied from the model workspace; policy.py is supplied atomically.
```

The host checks syntax, import, Policy(contract) construction and act(observation)
signature in a fresh read-only sandbox, under the existing response and phase time
limits. It never invokes act or physical feedback during this check. Validation
errors do not modify the workspace or select another candidate; the model may use
its remaining submission allowance to correct and submit again. Passing means
loadable interface, not legal actions for all inputs or successful manipulation.

The old v1.2 runs remain stopped without submission. Never load their sessions with
this release or retroactively freeze their last workspace file. The read-only console
now separates submission-stage remaining limits from unused global development
allowance, and shows failed submission packages as submission_attempt snapshots.

Submission-phase responses must contain exactly one submit call. Multiple candidate
calls are rejected without executing any; only an already-budgeted remaining call
can correct the response. The initial v1.3 candidate and synthetic smoke are retained;
v1.3.1 adds this unambiguous single-package delivery rule. Neither ran a live model.


## v1.3.2 nested sensor projection

`planar-option-json-v1.1` projects nested kinematics, scans and surface patches by
explicit field lists. `scan.raw_labels` is diagnostic-only and is no longer sent.
In the current scanner its values duplicate the public semantic category; no
additional object pose, map or obstacle ID was found in existing run records.
The change closes an accidental diagnostic-field export and prevents future nested
fields from silently expanding the observation. Dynamics, masks, options, budgets,
public sensing assumptions and success definitions remain the same. Keep old locks
and source snapshots; historical runs use their own snapshot for verification.

An external Codex relay may use the same driver and experimental limits, but its
relay exchanges and output reservations do not meter native Codex sampling,
reasoning, tokens or costs. Record this distinction in the campaign preflight and
report. A fresh fork avoids conversation contamination; relay-only author access is
an instruction, not OS isolation of the Codex author. Generated scripts and frozen
policies retain the existing hard process/filesystem/network sandbox.


## v1.3.3 declared thinking settings by stage

`flash_high_config.json` enables thinking with reasoning_effort=high during
development. Its explicit submission_settings disables thinking with effort=none
only in the reserved submission phase, because the provider rejects forced tool
choice in thinking mode. Existing non-thinking configs behave as before; no mode
is silently changed, and early voluntary submissions may still occur in high mode.
Exact effective settings are recorded in every model request. Reasoning continuation
fields are passed back for tool protocol continuity and redacted in the console.
The native preflight uses auto tool choice for high mode, forced choice otherwise.

This release changes only phase-aware model settings/preflight, tests and docs.
Observation, physics, policy decision timing, options, masks and success criteria
remain unchanged from v1.3.2. The high campaign keeps the same finite limits,
including 8192 requested output tokens per call and a 120 s request deadline;
a thinking/output truncation or timeout is an explicit failure, not permission to
silently increase budgets. Provider temperature has no effect in thinking mode.
See https://api-docs.deepseek.com/api/create-chat-completion/ and
https://api-docs.deepseek.com/guides/thinking_mode/ (checked 2026-11-09).


## v1.3.4 output truncation uses the reserved submission phase

A native response with finish_reason=length is charged and its provider usage is
recorded before finish-status validation. No partial tool call is executed or
installed. With a declared finalization reserve, development closes and the next
request uses only that existing submission allowance; no total cap is increased.
The partial response stays in raw events but is not added to conversational tool
history. A truncated submission consumes an attempt and may use only a remaining
reserved attempt. Transport failures still stop without retry.

The real v1.3.3 high attempt hit an 8192-token response cap without submission;
it is preserved as incomplete with no evaluation. Never resume it under v1.3.4
or treat its last workspace file as an implicit policy. This fix is regression-
tested; a new live campaign is separately reported, not inferred from those tests.


## v1.3.5 wide Flash high budget

At the user's request, flash_high_config.json now allows 131072 output tokens
per request, 1200 seconds per request and 7200 seconds total, still at most 40
requests (38 development + 2 reserved submission). Total output reservation is
5242880 tokens; cumulative input-byte cap is 128 MiB. Submission reserves 262144
output tokens, 16 MiB input and 1800 seconds inside those totals. These are caps,
not actual usage or a claim that all budgets must be spent. Physical experiment
limits, time-based decision interface, options and evaluation are unchanged.

The HTTP worker now receives the actual request deadline rather than retaining
a hardcoded 120-second socket timeout. The bounded response read is 16 MiB instead
of 2 MiB to accommodate longer reasoning. Outer process deadlines and no-retry,
no-redirect, credential isolation and redacted errors remain enforced. Output
truncation uses only the v1.3.4 predeclared submission reserve. Older snapshots and
attempts retain their original configuration; no automatic resume or policy choice.

## v1.3.6 missing normal-command initialization

ContinuousPush now initializes its per-step radial command to zero before branches
that may have no observed face or only brake. This prevents UnboundLocalError when
recording execution feedback and preserves existing loss-of-face and braking exits.
Four regression cases cover translation, both rotation signs and braking after a
previous nonzero diagnostic command. No action menu, sensor, score or policy changes.

The November 10 controller-repair reevaluation uses unchanged v1.3.5 submitted
packages and the same complete 30-case panel for each condition. Its host-only
runner records original submission provenance separately from the v1.3.6 execution
release, verifies both, and writes new evaluation directories. It does not relabel
an old submission as newly synthesized, change an old manifest, or bypass the
normal evaluator's frozen-release gate. Historical runs use their matching snapshots.

## v1.4.0 independent prediction profile (2026-11-10)

Optional config `prediction: {version: "nominal-v1", runtime: boolean}` exposes
an independent, bounded predictor in coding scripts and, when enabled, through
nested RPC inside Policy.act. See [API](prediction_api.md). Hypotheses are authored
from legal observations; the service never receives the execution environment.
Full synthetic state is explicitly labelled. No dt adjustment in nominal-v1.

Tests preserve old profiles without prediction. Runner now checks physical tick
exhaustion before requesting another policy action (previously it requested one
unused final action). Physical controller, observations, reward and success rules
are unchanged. Old releases/snapshots remain the authority for historical runs.

New paired campaign: `python -m experiments.planar_tasks.prediction_sol_20261110.campaign prepare`,
then `... campaign development` or `... campaign runtime`; these are explicit
bounded experiments, not automatic tests. A fresh Sol High author accesses only
that campaign's relay. Never reuse an old output directory or expose evaluation
artifacts to an author. Execution is paused while planning; latency is recorded.
