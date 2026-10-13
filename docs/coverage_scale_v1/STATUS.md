# STATUS - coverage_scale_v1

run_id `coverage-scale-20261013`; wall clock since the round started: **1.71 h** of an 8 h hard limit; reported at 2026-10-13 04:12:44 CST.

## Collection

| split | finished shards | windows |
|---|---:|---:|
| train | 291 | 139465 |
| val | 8 | 3852 |
| test | 8 | 3760 |
| narrow | 48 | 23071 |

## Training (finished cells)

- tier 8000: 6/6 cells - 8000_256_seed0, 8000_256_seed1, 8000_256_seed2, 8000_512_seed0, 8000_512_seed1, 8000_512_seed2
- tier 16000: 6/6 cells - 16000_256_seed0, 16000_256_seed1, 16000_256_seed2, 16000_512_seed0, 16000_512_seed1, 16000_512_seed2
- tier 32000: 6/6 cells - 32000_256_seed0, 32000_256_seed1, 32000_256_seed2, 32000_512_seed0, 32000_512_seed1, 32000_512_seed2
- tier 64000: 6/6 cells - 64000_256_seed0, 64000_256_seed1, 64000_256_seed2, 64000_512_seed0, 64000_512_seed1, 64000_512_seed2
- tier 128000: 6/6 cells - 128000_256_seed0, 128000_256_seed1, 128000_256_seed2, 128000_512_seed0, 128000_512_seed1, 128000_512_seed2
- narrow control: 6/6 cells - 16017_256_seed0, 16017_256_seed1, 16017_256_seed2, 16017_512_seed0, 16017_512_seed1, 16017_512_seed2

active coverage_scale processes: 1

## Deliverables

- contract `docs/coverage_scale_v1/CONTRACT.md`; errata `docs/coverage_scale_v1/errata.md`
- report `docs/coverage_scale_v1/report.md`; tables `docs/coverage_scale_v1/tables.md`;
  coverage `docs/coverage_scale_v1/coverage_table.md`; manifest `docs/coverage_scale_v1/manifest.json`
- raw shards, root registries and every checkpoint under this run directory
- generator/collector/trainer/evaluator entries under `experiments/coverage_scale/`, library
  under `src/coimagine/coverage_scale/`, acceptance tests `tests/test_coverage_scale.py`
- review bundle `coverage_scale_v1_review.zip`

## Recovery

```bash
python experiments/coverage_scale/collect.py --run-id coverage-scale-20261013   # skips finished shards
python experiments/coverage_scale/collect_narrow.py --run-id coverage-scale-20261013 --target-windows 16017 --shards 48
python experiments/coverage_scale/run_matrix.py --run-id coverage-scale-20261013 --tier 128000 --parallel 3
python experiments/coverage_scale/run_matrix.py --run-id coverage-scale-20261013 --tier 16017 --split narrow --parallel 3
python experiments/coverage_scale/run_decision.py --tiers 8000,16000,32000,64000,128000 --narrow-tier 16017
python experiments/coverage_scale/run_holdout.py --tiers 8000,16000,32000,64000,128000 --narrow-tier 16017
python experiments/coverage_scale/run_candidates.py --tiers 8000,16000,128000 --narrow-tier 16017
```

## Next step (one only)

Keep every frozen condition and run the next tier at a 256k cap with the same six runs and the
same new val/test. P is still falling at 128k and the C3 rule never asked to stop, so the round
ended on its frozen scope limit, not on a measured plateau.

