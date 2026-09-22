# `_old/` — deprecated Gate A (v4/v5) research line

Everything under this directory belongs to the **previous** CoImagine research line:
*"Distributed local-belief imagination of evolving coordination topologies"* (Gate A,
versions 0.3–0.14).

It is archived, not deleted, for two reasons:

1. the exact baseline is reachable through the git tag **`gateA-v0.9.1`**, so the code
   history is not lost;
2. a few low-level geometry helpers may be ported forward deliberately, with
   attribution, when the v6 2D grid stage needs them.

## Why it is deprecated

The v6 plan and the 2D execution prompt (`docs/CoImagine_2D_Agent_执行提示_v1.md`)
require a different core:

| | Gate A (this archive) | v6 2D first round |
|---|---|---|
| world | deterministic geometry | hidden physical type θ |
| observation | deterministic ray cast | joint likelihood `P(o_A,o_B,y_pub∣s,a,s')` |
| belief | 3-valued symbolic set + optimistic closure | exact Bayes posterior + private history |
| communication | none | one `B`-bit message per robot per boundary |
| planning | max over admitted actions, single rollout | `O_full/O_central/O_codec/O_B` with future branches |

Mixing the two would entangle two different claims, so the new core is written fresh
under `src/coimagine/`.

## Do not import from here

`_old/` is **not** on the package path: `pyproject.toml` discovers packages under
`src/` only. Nothing in the active tree may import `_old.*`. If a helper is worth
reusing, copy it into `src/coimagine/` and note the origin in a comment.

## Layout

```
_old/gate_a/
├── src/coimagine/     # full old package (topology, planning, belief, evaluation, reporting, envs, ...)
├── tests/             # 36 old tests
├── scripts/           # gate_a runners and smoke scripts
├── configs/           # gate_a_*.yaml profiles
├── docs/              # gate_a_*.md design notes
├── outputs/           # old run artifacts (gitignored)
├── README.md          # old project README
└── CHANGELOG.md       # old changelog up to v0.9.1
```
