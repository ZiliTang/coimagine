# Gate-A v0.5 pilot review guide

1. Run `pytest`; all semantic tests must pass.
2. Run `python scripts/run_gate_a_horizon_study.py --seed-stop 24`.
3. Confirm that open variants show IET/privileged-oracle root-action agreement and
   that CST generates zero new option templates.
4. Confirm that H=3 completes the open Reposition–Approach–Push tasks while H=2
   does not.
5. Inspect surprise variants for optimistic false positives and wasted
   repositioning; do not hide these failures.
6. Run `coimagine-gatea --profile smoke` and require zero pipeline failures.
7. Inspect the main storyboard and horizon figure before increasing the sample
   count with the paper profile.

Pilot values validate semantics and the selected horizon. They are not final
paper statistics until the procedural distribution and all budgets are frozen.
