# Gate A v0.9: heterogeneous randomized channel benchmark

The v0.8 causal families remain controlled mechanism tests. Version 0.9 adds a
separate generalization suite so aggregate results are not forced into deterministic
0/100 outcomes by construction.

Each scene samples map size, channel position, 2--4 blockers, blocker length and
coalition requirement, task weights, robot initial poses, sensing radius, sparse
reposition anchors, static occluders, optional activation dependencies, candidate
budget, and episode budget. No planner outcome is used during generation.

The debug gallery is written to
`figures/fig_gate_a_randomized_channel_debug.*`. It is intended for diagnosing
scene diversity before launching a large run. The quantitative summary reports
completion with Wilson intervals, median/IQR final task potential, and median/IQR
planning time by easy/moderate/hard strata.

The default `main` profile evaluates CST H=3 and IET H=3. The `full` profile also
adds H=2 controls and the privileged H=3 oracle, but is considerably more expensive.

```bash
python scripts/run_gate_a_horizon_study.py \
  --skip-controlled \
  --random-channel-per-difficulty 100 \
  --random-channel-candidate-budget 5 \
  --random-channel-max-steps-cap 6 \
  --random-channel-profile main \
  --jobs 8 \
  --output outputs/gate_a/v09_randomized_full
```

Exact H=3 enumeration becomes expensive as multiple entities compete for a fixed
candidate budget. The sandbox package therefore includes only a one-case-per-stratum
smoke result; paper statistics should be generated on the target workstation.

## Command-line progress

Long runs show live progress on stderr by default. Interactive terminals receive a
single updating bar; redirected logs receive periodic lines with completed cases,
generated episodes, throughput, elapsed time, and ETA. Use `--no-progress` to
disable it or `--progress-interval 10` to change the redirected-log interval.
