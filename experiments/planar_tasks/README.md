> 2026-11-11 起，新 CAP 实验默认使用 [authoring_v2/README.md](authoring_v2/README.md)：所有 LLM、自由 CAP 与结构化搜索均获得同一份[公开技能行为说明](authoring/option_guide.md)。v2 支持批量实验、实际成本结算、摘要优先及证据回查。旧冻结基线入口 [baseline/README.md](baseline/README.md) 保留复现用途，最小研究测台见 [console/README.md](console/README.md)。下面保留历史任务/旧 pilot 用法。

# Planar development tasks v1

**Latest baseline infrastructure:** [Autonomous baseline v1.1](baseline/README.md) /
[normative definition](../../docs/research/Agentic_Options_Baseline_v1.md), frozen
2026-11-09. The three-candidate workflow below is the historical
**Legacy Fixed-Feedback Pilot**, not the new autonomous baseline. Historical
programs/results remain intact. No new model experiment accompanied this freeze.

2026-11-09. User authorized direct implementation of the minimal task set and
baseline interface. On the same date the user explicitly authorized autonomous
coding-agent exploration and DeepSeek API use. The bounded pilot below has now
run; this remains development infrastructure, not a frozen benchmark or official
AgenticGenPlan reproduction. No RL training was started.

## Tasks and scope

`tasks.task_catalog()` defines nine scenes: open transport, near-wall release,
and transport around a fixed obstacle, each with anchor/mirror/shift variants.
All are **development**. Mirrors and shifts share their anchor's `root_id`;
they are not independent test examples. Do not derive a held-out test by moving
these variants to another split after inspecting results.

The original 1.0 × 0.25 m object, robot geometry, physics, V2 controllers,
90 s task horizon, reward and success/stopping criteria are retained. The wall
anchor has a 0.01 m initial object–wall gap. It represents the mechanism in the
user's drawing using the existing rectangle, not an exact reconstruction of its
aspect ratio. Rotation is a candidate solution; sliding or other lawful solutions
also count. The obstacle task allows either route; no straight-path requirement.

The geometric audit checks actual obstacle polygons, crossing-edge overlap,
robot/object/boundary clearance and a collision-free goal placement. It does
**not** prove dynamic reachability or that two robots are necessary. Every task
remains `feasibility=unverified` until a complete feasible execution is recorded.
Multiple movable objects, obstacle relocation and multi-goal tasks are deferred;
the current engine still has one movable task object and static obstacles.

```bash
.venv/bin/python -m experiments.planar_tasks.cli validate
.venv/bin/python -m experiments.planar_tasks.cli export --out outputs/planar_tasks/my_catalog
```

Exports are immutable new directories: `catalog.json` (including task hashes),
`policy_contract.json` and `task_atlas.png`. The existing 8765 testbed loads these
development scenes as `seed_planar_*` presets, shows the static obstacles and
rejects edited planar initial states that overlap obstacles. Historical presets
and replay files are not rewritten.

## Policy contract

`planar-option-json-v1` is a **centralized shared-sensor** interface. Both robots'
own odometry, scans, visible patches, contact readings, goal, arena and clock
are shared, with a whitelist of controller execution feedback. The policy may
retain its entire previous transcript. It does not receive object truth,
friction labels, scene ID, task family, full obstacle map, simulator objects,
reward, terminal goal distance or offline replay metadata. Nonfinite sensor
values serialize as `null`; a mask is always derived from legal observations.

The first JSONL request contains `reset: true`, `contract` and `observation`.
Subsequent requests contain `observation`. One response per request is exactly:

```json
{"a0": 36, "a1": 36, "ticks": 20}
```

`a0/a1` must be a pair in that request's `legal_pairs`. `ticks` is an integer
1–40 (0.05 s/tick). Repeated commands continue a persistent controller; they
do not reset a failed option. No automatic correction or hidden fallback is
performed for invalid responses. The client process persists within an episode
and is freshly started for the next episode.

The interface retains V2 RL's translation/reposition/BRAKE/YIELD combinations
and additionally exposes same-sign paired ROTATE through the existing coordinator.
FOLLOW and mixed rotation combinations are not offered in this first contract.
The V2 **RL event interface still excludes rotation** and its checkpoints remain
unchanged. Thus the broad option interface and the trained RL interface are
different action spaces; do not compare their scores as an equal-action experiment.

`JsonlPolicy` bounds reading/writing, handles partial lines and enforces a response
deadline. It is **not a security sandbox**: a local child can still read the local
filesystem. Run only reviewed programs here. A blind coding-agent study requires
separate source/test/filesystem/network isolation; IPC alone does not establish it.

## One bounded run

```bash
.venv/bin/python -m experiments.planar_tasks.cli run \
  --task planar_wall_anchor --out outputs/planar_tasks/my_run \
  --max-decisions 5 --wall-seconds 30 --response-seconds 5 \
  --publish-testbed \
  --command .venv/bin/python -m experiments.planar_tasks.policy --rotate-smoke
```

The included BRAKE and fixed-CW fixtures only check the protocol. Neither is a
research baseline, a learned method or an AgenticGenPlan-generated program.
Replace the command after `--command` with a reviewed policy's JSONL executable.
The executable must flush one JSON object per line; send debugging to stderr.
The episode runner never calls an LLM. The separate, explicitly invoked
`deepseek_agent` module calls the API only during policy synthesis; deployment
uses the frozen Python program. Model settings and usage are recorded.

The runner records source snapshots and hashes, Git HEAD/dirty state, configuration,
task hash, budgets, policy command, explicit command-file hashes, the policy
transcript (`sensors.jsonl`), stderr, executed decisions, truth-only diagnostics,
summary and a testbed-compatible replay. Transitive external dependencies/model
weights are not automatically archived. Physics advances only for executed ticks;
waiting for the policy does not advance simulation. Both wall and simulated times
are reported. Invalid output, process failure, wall/decision limits and numerical
failures remain in the attempted-episode denominator. Incomplete runs exit with
code 2, retaining evidence; the runner never retries or silently adds budget.

The reporting metric `task_cost_seconds` is explicitly separate from the unchanged
environment reward: successful attempts pay simulated elapsed time; every failed
or incomplete attempt pays 180 s (90 s horizon + 90 s failure charge). This fixed
failure convention prevents a crash or early resource cutoff from looking cheap.
It is a provisional development metric, not a retrospective change to RL scores.
Freeze evaluation budgets and metrics before any comparative experiment.

`--publish-testbed` explicitly copies only the replay into local testbed history
under a fresh ID and prints its URL. It never overwrites an old run. Replay
frames show post-tick truth while contact/scan entries describe that tick's
pre-execution sensing; truth files are never fed to a policy.

## Baseline attribution and remaining work

The intended first strong baseline follows the **interactive program synthesis**
workflow in Merler et al., *Coding Agents for Generalized Task and Motion Planning
Problems*, [arXiv:2609.30233v1](https://arxiv.org/abs/2609.30233v1), 2026.
As clarified by the user, treat it as an **arXiv preprint**, not an accepted or
published ICRA paper. The arXiv page was checked on 2026-11-09. A preprint can
be cited and evaluated; distinguish its publication status from the strength
of its experimental evidence.

Use a name such as **AgenticGenPlan-style / CoImagine-Options (our adaptation)**.
Do not label this scaffold an official implementation or an exact reproduction.
The original setting uses fully observed object-centric states and different
action interfaces; our version supplies partial sensor evidence and handcrafted
options. Document these changes, development interaction/model budgets,
evaluation horizon and observations, independent synthesis seeds, and test splits.
Pin paper version, upstream code commit (if used), prompt and model version.
If code is reused, check its license and retain attribution.

The supplied local PDF differs from the public experimental version. Its SHA256:
`7440db67d95406bdfa9239bc75b0b360b3cae3abe93acfb5519f4bc83437304a`.
Do not mix experimental numbers across those documents. Suggested citation entry:

```bibtex
@misc{merler2026codingagents,
  title={Coding Agents for Generalized Task and Motion Planning Problems},
  author={Merler, Matteo and Li, Bowen and Roy, Josh and Liang, Yichao and
          Wang, Qianwei and Huang, Yixuan and Silver, Tom},
  year={2026},
  eprint={2609.30233},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2609.30233v1},
  note={Preprint; version 1}
}
```

Next: review the wall anchor's complete behavior with the user, establish at least
one executable solution under the fixed skills, then fix a coding-agent development
budget and isolation mechanism before synthesizing policies. No agent synthesis,
LLM/API benchmark, full episode feasibility proof or independent RL test has yet
been performed by this implementation. The common failure categories to record
are skill execution, missing observable evidence, wrong skill/parameter choice,
subgoal ordering and failure to recover; diagnose rather than assume a memory gap.


## Bounded coding-agent workflow (2026-11-09)

`agentic.py` implements generate → execute development trials → sensor feedback →
repair → freeze → development transfer. This is a restricted program-synthesis
adaptation of the cited preprint, not a full reimplementation of its tool suite.
There is no online LLM planner or experience retrieval in this pilot.

Three original authors: this GPT-6 Astra session (repository-informed), a GPT-6.1
Sol subagent (packet/feedback-only by instruction), and DeepSeek-V4-Pro API
(packet/feedback only, thinking disabled). Each original session allowed three
programs and six development episodes, followed by seven frozen transfer episodes.
All nine scenes are still development fixtures. Different source access, model
settings and unmetered in-session tokens prevent a fair model ranking.

```bash
.venv/bin/python -m experiments.planar_tasks.agentic init \
  --session outputs/agentic/my_deepseek_session --model deepseek-v4-pro \
  --access 'API packet and feedback only' --maximum-candidates 3
# Explicitly makes up to three billable calls; no automatic retry/resumption:
.venv/bin/python -m experiments.planar_tasks.deepseek_agent \
  --session outputs/agentic/my_deepseek_session
.venv/bin/python -m experiments.planar_tasks.agentic transfer \
  --session outputs/agentic/my_deepseek_session
```

Set `DEEPSEEK_API_KEY` outside the repository, or use
`~/.config/coimagine/deepseek_api_key` with mode 0600. Credentials are read only in
the API request worker. They are never passed to generated policies, command-line
arguments, source snapshots or transcripts. The client uses the fixed official
HTTPS endpoint, refuses redirects, caps calls/output/request size and enforces
180 s per API call in a separate worker. Recorded usage is provider-reported token
usage, not an invented monetary estimate. An interrupted/billable attempt is not
retried automatically.

For an in-session author, use `agentic trial --session ... --policy policy.py
--note 'rationale'`, up to the declared candidate limit, then `agentic freeze`.
Selection is fixed before trials: most successes, least total task cost, least
terminal distance, earliest candidate. Syntax errors consume a candidate; run
errors remain unsuccessful episodes. Frozen hash checks reject later edits.
`transfer` is separate, never repairs a program, and refuses overwriting a run.

Generated deployments run through `sandbox.py`: bubblewrap removes host network,
repository, home directory and inherited environment. Only standard-library
Python, the read-only submitted file and ephemeral `/tmp` are available. Memory,
CPU, file size, JSONL I/O and response deadlines are bounded. No unsandboxed
fallback exists. This does not sandbox the Codex synthesis author itself.

Every session retains its exact packet, model/access setting, candidate programs,
rationales, episode source snapshots, legal sensor/action transcript, development
feedback, truth/replays (evaluation-only), freeze hash and transfer results. API
sessions additionally retain sanitized request/response JSON and usage. In-session
Astra/Sol author reasoning/token costs are not exported API transcripts and are not
claimed as such.

Original DeepSeek programs misread the request envelope and compared tuple actions
to JSON lists, yielding only BRAKE. Those failed runs are preserved. A separate
one-call assisted protocol repair uses explicit wire examples and public settling
criteria; it is not pooled into the original experiment. The packet version records
this clarification. See the [pilot review](../../docs/agent_exchange/reviews/Agentic_Pilot_20261109.md)
for actual results, artifact hashes, replay links and limitations.
