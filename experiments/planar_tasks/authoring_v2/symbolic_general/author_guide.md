# Symbolic CAP general v1: author contract

Author a causal policy as a literal `SYMBOLIC_POLICY` in the generated Python
carrier. Read the normal authoring_v2 guides, current relational option supplement,
and checked_policy_guide/example. Current geometry is compact20, 72 rays, 45-degree
visible-face PUSH; REPOSITION never automatically acquires the next PUSH contact.
Prefer several short actions over one long action. All task costs and success
conditions remain the existing ones.

The host exposes `frontend.semantic_packet(memory, public_observation)`: current
target observation, estimated goal vector/distance, geometry uncertainty, elapsed
stagnation, contacts, speed, eight goal-direction alignments, estimated directional
wall gaps and whether a wall was actually observed, and per-option menu legality.
The current gap estimate is not a guaranteed lower bound. `permit:OPTION` is an
epistemic permission from the frozen effect model or online public-hypothesis rollout;
its absence defaults true during unguarded development and successor model inquiry.
Every execution rule MUST include its `legal:OPTION` and `permit:OPTION` predicates.
The host checks this logically before author acceptance. Do not remove these gates.

Available options are listed by `frontend.option_table()`: eight PUSH headings,
dual push or single pusher + YIELD, 6/12 ticks; each robot CW/CCW orbit + YIELD for
12 ticks; eight target-free paired MOVE headings at 12 ticks; BRAKE at 4 ticks.
You choose these actions using your own editable rules and semantic expressions.

## Declaration

`version:1`, `derived:{name:expression}`, `rules:[{name,when,option,score}]`,
`expect:{option:post_expression}`. Derived definitions are topologically ordered.
Highest score among enabled rules wins; earlier rule breaks equal-score ties.
The score is dimensionless. `expect` declares promised post-semantic properties,
not new task rewards. These commitments are checked on forecasts and measured data.
Physical task termination is host-owned. BRAKE is safe-stop intent, never productive
coverage or a guaranteed frozen physical world.

Expressions: `["var",name]`, `["lit",value,unit]`, `and/or/not`, `lt/le/gt/ge/eq/ne`,
same-unit `add/sub/min/max`, `abs/neg`, dimensionless scaling `mul`, division by a
nonzero dimensionless literal, and `if` with equal-unit branches. Units are `m`,
`s`, `m/s`, `1`, `bool`. Example:

```python
{"name":"short_NE", "option":"push_both_1_6",
 "when":["and",["var","legal:push_both_1_6"],
                 ["var","permit:push_both_1_6"],
                 ["gt",["var","align_1"],["lit",0.8,"1"]]],
 "score":["var","align_1"]}
```

The checker predicts next primitive summaries, reevaluates these exact selection
rules, and checks the ACTUALLY selected next option's precondition. Missing written
predictions become unknown. Joint outcome samples remain paired. Empirical model
admission is conditional, not a proof of physical-world coverage. A forecast of a
visible face supporting the desired PUSH is an effect of orbiting worth checking.

## Tools (repository root, `.venv/bin/python`)

Write a JSON rule declaration under the supplied author directory. Generate its
complete source: `-m tools.cap_symbolic_general build --rules PATH --out PATH`.
Use `-m tools.cap_symbolic_general status` for the clock and metered budgets.
Use `... test --policy PATH --rooms 4` (1..16 declared author rooms) for complete
1800-tick development execution. `trace --trial N --room ID --start N --count 5`
returns public semantics/actions and labeled development physical effects.
`probe --policy PATH --trial N --room ID --boundary N --actions OPTION ...`
replays identical prefixes, tries up to eight alternatives, and actually executes
the next action selected by your policy. Replay and candidate ticks are charged.

`fit` fits a public-semantic effect model from your collected development records.
`diagnose --policy PATH --trial N --room ID --boundary N --model PATH` checks
the resulting predictions; `--repair` automatically collects missing evidence for
the proposed option and same-control alternative duration, then fits a new model.
No query may access fitting/calibration/confirmation room fixtures or outcomes.
Only the `author/` directory, supplied source guides, and these tools are allowed
research inputs. The final evaluator uses different rooms.

The wall budget is 3600 seconds including a final 60-second submit-only reserve,
250000 real physics ticks including prefixes, and 2000 resets. Failures consume
budget; unfinished reservations are not refunded. No new API calls or extra agents.
The host's later multi-version model study is separate from this author budget.

## Final submission

Use `tools.cap_policy_check lint`, then `submit --policy PATH --boundary INPUT ...
--out NEW_DIRECTORY` on the three supplied host boundaries; `verify --submission
DIRECTORY`. Keep the generated carrier unchanged except the literal declaration.
Write `FINAL.json` with `policy`, `checked_submission`, `development_evidence`,
`limitations`. Explicitly choose your program before the deadline. You may finish
early if you have a tested useful policy; a full hour is a cap, not a minimum.
