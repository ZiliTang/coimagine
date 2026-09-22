# Gate A v0.7: Mixed Randomized Quantitative Benchmark

The v0.7 study removes the binary all-success/all-failure artifact of the original
single-chain generator. Seeds are assigned without outcome screening to four
balanced causal families:

1. direct coordinated Push;
2. grounded Approach -> Push;
3. mixed contact/setup -> Push;
4. Reposition -> Approach(new contact) -> coordinated Push.

Map translation, horizontal reflection, and robot initial positions vary across
seeds. The first three families remain solvable by CST with H=2; the fourth
requires future observation-support expansion and H=3. Half of the fourth-family
instances contain an unseen support-wall surprise to quantify optimistic false
options and topology edges.

The quantitative script exports:

- episode-level results;
- aggregate H=2/H=3 completion and timing;
- completion by causal dependency class;
- successor option/topology fidelity and optimism cost;
- entity-count and robot-count structural scaling curves.

Run:

```bash
python scripts/run_gate_a_horizon_study.py \
  --seed-start 0 --seed-stop 80 \
  --output outputs/gate_a/horizon_v07_mixed
```
