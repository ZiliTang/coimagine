# local_contact_option_v1 - summary tables

test executions: 3840; dev executions: 768

## Layer 1 - execution clarity and physical validity

| arm | exits | effective 2s (pushable, baseline) | contact ratio | saturation | max penetration | settled observed | false stable | final speed median |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F | wall_clock_end:960 | 0.000 | 0.268 | 0.000 | 0.0011 | - | - | 0.0375 |
| N | contact_lost:60, face_switch:3, settled_observed:897 | 0.925 | 0.567 | 0.000 | 0.0012 | 0.934 | 0.000 | 0.0104 |
| O | wall_clock_end:960 | 0.000 | 0.467 | 0.000 | 0.0015 | - | - | 0.1591 |
| P | contact_lost:89, settle_unobserved:2, settled_observed:869 | 0.925 | 0.597 | 0.000 | 0.0015 | 0.905 | 0.035 | 0.0171 |

## Settle cross table (must not be omitted)

| arm | observed & truly settled | observed but still moving (false stable) | unobserved & settled | unobserved & moving |
|---|---:|---:|---:|---:|
| N | 897 | 0 | 63 | 0 |
| P | 839 | 30 | 66 | 25 |

## Layer 2 - perturbation sensitivity of the final state (object, m)

| arm | pos 2mm P90 | pos 10mm P90 | vel 0.02 P90 | vel 0.05 P90 | angle 10mm P90 |
|---|---:|---:|---:|---:|---:|
| F | 0.0036 | 0.0172 | 0.0155 | 0.0315 | 0.0268 |
| N | 0.0018 | 0.0056 | 0.0101 | 0.0239 | 0.0216 |
| O | 0.0079 | 0.0300 | 0.0259 | 0.0638 | 0.0516 |
| P | 0.0099 | 0.0126 | 0.0264 | 0.0710 | 0.0180 |

## Layer 3 - constant body-frame twist fit (PUSH trajectories only)

| arm | push completed | aborted before completion | twist fitted | fit pos RMSE | fit angle RMSE | extrapolation pos RMSE | straight-line share |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 0 | 0 | 0 | - | - | - | - |
| N | 897 | 63 | 882 | 0.0069 | 0.0037 | 0.0231 | 0.757 |
| O | 0 | 0 | 0 | - | - | - | - |
| P | 871 | 89 | 771 | 0.0050 | 0.0028 | 0.0273 | 0.462 |

## D.1 frozen engineering gates

| gate | value | threshold | pass |
|---|---:|---:|:--:|
| G2_effective_push_ratio_baseline_pushable | 0.9250 | 0.9 | yes |
| G3_false_stable_and_coverage | 0.0000 | 0.05 | yes |
| G4_displacement_not_much_smaller | 0.4371 | 0.8 | no |
| G5_perturbation_final_state | 0.5559 | 0.2 | yes |
