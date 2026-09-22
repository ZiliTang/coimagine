# AGENTS.md — working rules for the CoImagine repository

This repository implements the **2D validation** of the CoImagine v6 research plan.
The authoritative documents are:

1. `docs/CoImagine_2D_Agent_执行提示_v1.md` — **defines the current round's scope and
   acceptance criteria. Follow it over anything else.**
2. `docs/CoImagine_v6_研究方案与创新评估.md` — long-term research background.
   Do **not** implement the whole v6 plan in one go.
3. `docs/2d_protocol_spec.md` — the frozen state / observation / timing / skill /
   cost / oracle protocol for this round.

The previous Gate A research line (v0.3–v0.14, "distributed local-belief imagination
of evolving coordination topologies") is archived under `_old/`. It is reachable via
the git tag `gateA-v0.9.1`. See `_old/README.md`.

---

## 1. Scope of the current round

In scope: 2D only. A micro finite-state diagnostic (M0–M3) with exact tabular
generative models, exact Bayes beliefs, `B`-bit messages, and the oracle hierarchy
`O_full / O_central / O_codec / O_B / TrueModelLimitedSearch`; then a small 6×6 grid
pushing stage reusing geometric primitives.

Out of scope — do **not** start these without an explicit instruction:

- 3D simulation,双臂辅助仿真, mobile-robot hardware;
- neural world models, learned message encoders, any training loop;
- `option_prior` / general-model API calls (keep the interface, default off);
- deleting or rewriting the `_old/` archive.

## 2. Do not

- Fabricate experiments or results. Everything reported must come from a run.
- Call a heuristic or limited search an "exact oracle". Use `TrueModelLimitedSearch`
  and say it is not optimal.
- Modify reward/cost definitions or test conditions in order to obtain a positive
  result. Negative results are reported, not hidden.
- Add an information-gain reward, an "enabling" reward, or any bonus that manufactures
  the value of preparation/identification. Value must emerge from the transition and
  observation model under the unified task cost.
- Design a button labelled "probe" that reveals hidden truth for free. A probe is a
  real short skill; its displacement, cost and observation come from the same
  transition/observation tables as everything else.
- Prune push/probe candidates using hidden truth (resistance, friction, constraint
  mode). The candidate mask must be computable from public information alone,
  otherwise the mask itself leaks the answer.
- Pass `TrueState`, the environment object, or the teammate's private history to a
  deployment policy. Privileged data is for offline oracle design and evaluation only.
- Claim the whole first round is complete while quietly shrinking its scope. List
  unfinished items and the complexity reason.

## 3. Information boundary and timing

Every decision boundary follows this order, and the order is testable:

1. the environment distributes each side's **private** observation and the **same**
   common observation;
2. each side updates its belief using only its own history and the common record;
3. both sides rebuild the same candidate set and the same candidate IDs from public
   information;
4. each side sends **one** message of at most `B` bits. A message must not depend on
   the teammate's not-yet-received same-round message;
5. both sides update the common record; one deterministic public rule selects the
   joint option, with a fixed candidate order, seed and tie-break;
6. execute, charge cost, generate the next observations.

Rules that follow from this:

- `B = 0` means the public coordination policy may not secretly use either side's
  private belief. Do not claim this is optimal over all decentralised policies; the
  first-round oracle only covers the "one synchronous message round, then public
  commit" strategy class.
- With `B = 0`, do not describe the result as the optimum of the unrestricted
  Dec-POMDP.
- No future real feedback may be used before it is generated. Repeated forwarding of
  the same evidence must not be counted as an independent likelihood update.
- Keep `TrueState`, `PublicContext`, `PrivateHistory` and `OracleContext` as distinct
  data objects. A test that changes the hidden truth while holding the visible inputs
  fixed must leave ordinary rankers and policy inputs unchanged.

## 4. Probability, belief and observation model

- Use exact tables first. No neural world model in this round.
- Every robot performs a correct Bayes update under the shared generative model.
- **Never multiply two posteriors that share a prior** and call the result a joint
  belief. Public coordination generally needs `P(s, h_A, h_B | C)`, because messages
  depend on the private histories. Enumerate private histories in micro instances, or
  prove that the finite local statistic used is sufficient.
- Publish the complete joint observation table `P(o_A, o_B, y_pub | s, a, s')`.
  Do **not** assume the two sides' observations are independent unless it has been
  verified for that instance.
- Probabilities must be non-negative and normalised; impossible observations need an
  explicit treatment (zero likelihood must not produce NaN).

## 5. Cost and message accounting

- Primary cost is the cumulative action/time cost of reaching the goal, plus an
  explicit failure/truncation cost. Fix all terms in the configuration and state the
  units. A failure must not become cheap by exiting early.
- Also report success rate, executed steps, invalid pushes, probe count, bits,
  search nodes and wall-clock time.
- Serialise messages with an inspectable discrete codebook. Never label a Python
  float array as "1 bit". Candidate IDs, missing masks and similar content count
  toward the payload. Compare `B = 0, 1, 2` per side.
- A 1-bit message that transmits a fully binary observation is normal in a micro
  instance; report it. Do not artificially prevent a simple posterior from winning.

## 6. Oracle naming discipline

Implement and name separately:

| name | sees | use |
|---|---|---|
| `O_full` | true state, true dynamics, same finite horizon | full-information cost lower bound |
| `O_central` | both sides' available observation histories, not hidden truth | exact POMDP reference |
| `O_codec` | fixed message encoding rule, true model, exact planning incl. future observation branches through the same codec | separates message-representation loss from search error |
| `O_B` | same message budget and protocol class as deployment; enumerates local message rules and public action rules | fair optimal reference for tiny instances |
| `TrueModelLimitedSearch` | true model, node/depth limited | computation-limit diagnostic, **not** an optimal oracle |

For every exact result report the actual horizon, strategy class, whether the codec is
fixed, nodes explored and wall-clock time. If a budget is exhausted, return an explicit
incomplete status — never a pseudo-exact number.

When the instance, cost and horizon match, strategy classes are nested, and the
central method can simulate communication, verify:

```
J_full* <= J_central* <= J_B* <= J_codec*
```

Do not claim that using the true model is automatically better; limited search can
have a different bias.

## 7. Repository layout and commands

```
src/coimagine/
├── protocol/    # isolated data objects and the decision-boundary machinery
├── toy/         # micro finite-state generative models and instances M0-M3
├── exact/       # Bayes filters, codecs, exact oracles, limited search
├── diagnostics/ # runs the micro oracle suite, writes the gap table
└── cli.py
tests/           # property and protocol tests
docs/            # plans and the frozen protocol spec
_old/            # deprecated Gate A; never import from here
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest                      # full suite
pytest tests/test_oracles.py  # focused
python -m coimagine.cli --help
```

`_old/` is not on the package path and must never be imported.

## 8. Test requirements

At minimum, the suite must cover:

- probabilities non-negative and normalised; Bayes update agrees with a hand-worked
  micro example; impossible observations handled;
- the same public record yields the same joint action; a same-round message cannot be
  read before it is sent;
- future real feedback is not used before it exists; re-forwarded evidence is not
  counted twice;
- messages really are `B` bits; the oracle cost ordering holds where applicable;
- increasing `B` with nested strategy classes does not increase the exact `O_B` cost;
- collision / contact / combined-force / reposition cost / termination are correct;
- paired examples with identical public input and different private evidence show a
  changed decision, or explain why it does not change;
- an uninformative or too-expensive probe is not chosen by the exact policy, while a
  useful one may be chosen — the solver decides, not a hard-coded rule.

## 9. Stop point for this round

Stop after: micro oracles correct, the grid stage replayable, the minimal comparisons
actually run, and the diagnostic report written. Do **not** auto-expand to 3D, neural
model training, or a general-model API. Do not add new modules merely because no
method advantage appeared.
