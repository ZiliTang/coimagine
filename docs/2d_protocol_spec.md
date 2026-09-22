# CoImagine 2D — protocol specification (first round)

Status: **frozen for the first round**. Changes require a note in this file and a
matching test update. This is the `spec.md` required by
`docs/CoImagine_2D_Agent_执行提示_v1.md` §二 阶段0.

Scope: 2D only. Stage 1 = micro finite-state diagnostics M0–M3 with exact tabular
models. Stage 2 = 6×6 grid cooperative pushing. No learning, no neural models,
no general-model API in this round.

---

## 1. State and information

### 1.1 True state

```
s = (x, theta)
```

- `x` — **public configuration**: the coarse, shareable task/contact configuration.
  In the micro family `x ∈ {CONTACT, PREPARED, GOAL}`; in the grid it is the coarse
  object pose / occupancy known to both sides.
- `theta` — **hidden physical type**, stationary within an episode. In the micro
  family `theta ∈ {HEAVY, SLIP}` (or irrelevant for M0). In the grid
  `theta = (r, c, tau)` with resistance `r ∈ {1,2}`, constraint mode
  `c ∈ {free, constrained}` and local traction `tau`.

`x` is the part of the state that may be shared at the specified (low) frequency.
`theta` is never public. The simulator owns `s`.

### 1.2 Public context `C_t`

```
C_t = (task, coarse geometry, y_pub[0..t], a[0..t-1], m_A[0..t-1], m_B[0..t-1],
       candidate set + stable IDs, protocol version, timestamps)
```

- `y_pub` — common motion feedback at the specified frequency (grid: object moved /
  stalled; micro: `moved` / `stalled`).
- `a[t]` — the committed joint action at boundary `t`.
- `m_i[t]` — robot `i`'s message at boundary `t`.
- Candidate IDs are assigned by a deterministic public rule from public geometry.

### 1.3 Private history `h_i,t`

```
h_i,t = (o_i[0..t])
```

`o_i` is robot `i`'s **own** local contact/drive feedback (grid: own driving
response, local slip; micro: own binary `loaded` / `slipping` signal). `h_i` is never
shared except through `m_i`.

### 1.4 Joint observation model

The complete joint likelihood is published and queryable:

```
O( (o_A, o_B, y_pub) | s, a, s' )
```

- It is stored as an **explicit joint table**. Independence of `o_A` and `o_B` is
  **not** assumed; the micro instances deliberately use correlated tables.
- `y_pub` is a deterministic function of `(x, x')` in the micro family (moved iff the
  object advanced) and is stored in the same table so that all methods consume one
  object.
- Impossible observations have probability exactly `0`; the Bayes filter must handle
  a zero-likelihood event without producing `NaN` (it returns an explicit
  `ImpossibleObservation` status for that branch).

---

## 2. Timing of one decision boundary

Executed in exactly this order; every step is testable.

1. Environment draws `(o_A,t, o_B,t, y_pub,t) ~ O(· | s_{t-1}, a_{t-1}, s_t)` and gives
   `o_i,t` privately to robot `i`, `y_pub,t` to both.
2. Each robot updates its belief using **only** `h_i,t` and `C_t`.
3. Both sides rebuild the identical candidate set and candidate IDs from public
   information only.
4. Each side sends **one** message `m_i,t = mu_i(h_i,t, C_t)` with at most `B` bits.
   A message may not depend on the teammate's not-yet-received same-round message.
5. Both sides append to the common record; one deterministic public rule
   `pi(C_t, m_A,t, m_B,t)` selects the joint option. Candidate order, RNG seed and
   tie-break are fixed and identical on both sides.
6. Execute `a_t`, charge cost, generate `s_{t+1}` and continue.

`t = 0`: the environment **does** distribute an initial local observation
`(o_A,0, o_B,0, y_pub,0)`. A probe is therefore only valuable through the *additional*
information it produces, never as the first way to see anything.

---

## 3. Skills and actions

A joint action is `a = (a_A, a_B)`; every robot acts at every boundary.

Micro family:

| action | meaning | advance? | notes |
|---|---|---|---|
| `CO_PUSH` | both robots push the object together | depends on `theta`/`x` | cooperation action |
| `RELEASE_B` | B performs a lateral reposition/release | changes `x` | preparation, not information |
| `PROBE` | A performs one short diagnostic push while B holds | no | real short skill; displacement/cost/observation all defined by `T` and `O` |
| `HOLD` | both wait | no | zero cost, used for the too-expensive-probe negative control |

Grid family (stage 2): single push, same-direction cooperative push, one-robot-hold +
other-repositions, short diagnostic push, wait. A robot must occupy a legal contact
cell; the target region must be free; repositioning follows a collision-free grid path
and is charged its path duration (no free teleport). Coalition requirement and the
hidden resistance `r` are read from the transition table; the candidate generator may
only use public geometry.

**Prohibition.** No action may be added whose only role is to reveal truth, no
information-gain reward, no "enabling" reward. Probe and preparation values must fall
out of `T`, `O` and the unified cost.

---

## 4. Cost and termination

- `C(s, a, s')` is the per-transition cost. Units and values are fixed in the instance
  configuration. In the micro family: `CO_PUSH = 1.0`, `RELEASE_B = 1.0`,
  `PROBE = c_probe` (reported per instance), `HOLD = 0.0`.
- Episode cost = cumulative transition cost + `c_fail` if the horizon `H` is exhausted
  without reaching `GOAL`. `c_fail` is explicit and positive; **a failure must not be
  cheap by exiting early**.
- Success = `x_H == GOAL`. Reported separately: success rate, executed steps, invalid
  pushes, probe count, bits sent, search nodes, wall-clock time.
- Comparison across methods uses the same instances, cost and horizon.

---

## 5. Beliefs and the public coordination state

- Robot belief: `b_i,t(s) = P(s_t = s | h_i,t, C_t)` via exact Bayes under the shared
  generative model.
- **Public coordination state**, needed because future messages depend on private
  histories:
  `mu_t(s, h_A, h_B) = P(s_t, h_A,t, h_B,t | C_t)`.
  Micro instances enumerate the private histories exactly. Caching only `P(s | C)` is
  not sufficient and must not be used to predict future messages.
- Two robots' posteriors must **never** be multiplied as if independent. Where a
  joint quantity is needed it comes from `mu_t`, which is built from the joint
  observation table.

---

## 6. Messages and bits

- Message alphabet is the nested set `M_B = {0, ..., 2^B - 1}`; `B = 0, 1, 2` per
  robot per boundary.
- Every message is serialised with an inspectable discrete codebook. The bit count
  reported is the real codebook payload, including candidate IDs and missing masks.
  A Python float array is never reported as "1 bit".
- A micro instance in which one bit fully carries a binary local observation is a
  normal, reportable outcome and is not suppressed.

---

## 7. Oracles

| oracle | information | solution concept | role |
|---|---|---|---|
| `O_full` | true `s`, true model, horizon `H` | exact backward induction | full-information lower bound |
| `O_central` | both robots' observation histories, true model | exact POMDP / expectimax over the joint information state | complete-observation-sharing reference |
| `O_codec` | a fixed message encoding rule `mu`, true model | exact planning over the public policy whose future messages pass through the same `mu` | isolates message-representation loss |
| `O_B` | same `B`-bit budget and same protocol class as deployment | enumerate local message rules and public action rules | fair optimal reference for tiny instances |
| `TrueModelLimitedSearch` | true model, node/depth limited | e.g. myopic depth-1 or node-capped expectimax | computation-limit diagnostic, **not** optimal |

Every result reports: actual horizon, strategy class, whether the codec is fixed,
nodes explored, wall-clock time, and an explicit `incomplete` status if a budget was
exhausted. `O_B` is restricted to the strategy class "one synchronous message round,
then public commit"; it is **not** the unrestricted Dec-POMDP optimum.

When instance, cost and horizon match, strategy classes are nested, and the central
method can simulate communication, verify:

```
J_full* <= J_central* <= J_B* <= J_codec*
```

In the first delivery `O_B` is exact for a **single decision boundary** (`H = 1`).
Multi-step exact `O_B` is a Dec-POMDP and is reported as `incomplete` with the
complexity reason; a separately named restricted class may be reported but must not be
labelled `O_B`.

---

## 8. Micro instances M0–M3

Common micro tables: `x ∈ {CONTACT, PREPARED, GOAL}`, `theta ∈ {HEAVY, SLIP}`,
actions `{CO_PUSH, RELEASE_B, PROBE, HOLD}`, horizon `H = 3`, local observables
`o_i ∈ {loaded, slipping}`, public observable `y_pub ∈ {moved, stalled}`.
Exact enumeration size is reported per instance by the runner.

- **M0 — public information sufficient.** The public observation identifies the
  recovery action; private local signals carry no decision-relevant information about
  `theta`. Expectation: communication gives no gain (`J_B = J_codec = J_central`).
- **M1 — stall-cause ambiguity.** Public `stalled` is identical for both causes; the
  correlated joint local table makes `o_A` and `o_B` partial, complementary evidence.
  `HEAVY` succeeds with `CO_PUSH` from `CONTACT`; `SLIP` requires `RELEASE_B` first.
  Expectation: with `B = 0` the team cannot beat the prior-optimal action; a short
  message changes the committed action.
- **M2 — costly identification.** As M1, but the initial local observation is weakly
  informative and `PROBE` produces a sharp observation at cost `c_probe`. Two
  negative controls:
  - *M2-uninformative*: the probe observation is independent of `theta`; the exact
    policy must not probe.
  - *M2-expensive*: `c_probe` is large; the exact policy must not probe.
- **M3 — preparation changes the contact mode** (optional but included). `theta` is
  effectively known; `CO_PUSH` from `CONTACT` cannot succeed and `RELEASE_B` is
  physically required to reach `GOAL`. This separates physical enabling from
  information gathering: the value of `RELEASE_B` is not an information term.

`M0`, `M1`, `M2-uninformative`, `M2-expensive`, `M3` are all constructed from the same
engine; no instance-specific reward term is allowed.

---

## 9. Stage 2 grid contract (preliminary, implemented later)

- 6×6 map, two robots, one 2×2 object, obstacles, target region; translation only
  (a rotation interface is retained but not required).
- Public: geometry, committed actions, low-frequency object pose. Private: own
  drive/slip feedback.
- Hidden resistance `r ∈ {1,2}`; a single robot's push capacity is 1, same-direction
  cooperative capacity is 2.
- Candidate generation from public geometry only; hidden `r` must not prune candidates.
- Large grids use `TrueModelLimitedSearch` and are never labelled "exact optimal".

---

## 10. Configuration and reproducibility

- Instances, costs and message budgets are declared in code/config, not tuned on test
  outcomes.
- Seeds are fixed and recorded. Randomness that changes with the action sequence is
  not described as a fully paired observation trajectory.
- Results record the git revision, configuration, seeds and the exact command.
