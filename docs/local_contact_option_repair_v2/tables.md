# local_contact_option_v1 - summary tables

test executions: 3840; dev executions: 768

## Layer 1 - execution clarity and physical validity

| arm | exits | effective 2s (pushable, baseline) | contact ratio | saturation | max penetration | settled observed | false stable | final speed median |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F | wall_clock_end:960 | 0.000 | 0.253 | 0.000 | 0.0014 | - | - | 0.0355 |
| N | contact_lost:56, face_switch:7, settle_timeout:190, settle_unobserved:82, settled_observed:625 | 0.938 | 0.508 | 0.000 | 0.0013 | 0.651 | 0.000 | 0.0060 |
| O | wall_clock_end:960 | 0.000 | 0.467 | 0.000 | 0.0015 | - | - | 0.1591 |
| P | contact_lost:88, settle_timeout:191, settle_unobserved:116, settled_observed:565 | 0.925 | 0.561 | 0.000 | 0.0032 | 0.589 | 0.023 | 0.0006 |

## Settle cross table (must not be omitted)

| arm | observed & truly settled | observed but still moving (false stable) | unobserved & settled | unobserved & moving |
|---|---:|---:|---:|---:|
| N | 625 | 0 | 335 | 0 |
| P | 552 | 13 | 395 | 0 |

## Layer 2 - perturbation sensitivity of the final state (object, m)

| arm | pos 2mm P90 | pos 10mm P90 | vel 0.02 P90 | vel 0.05 P90 | angle 10mm P90 |
|---|---:|---:|---:|---:|---:|
| F | 0.0042 | 0.0178 | 0.0155 | 0.0316 | 0.0265 |
| N | 0.0029 | 0.0055 | 0.0097 | 0.0242 | 0.0215 |
| O | 0.0079 | 0.0300 | 0.0259 | 0.0638 | 0.0516 |
| P | 0.0089 | 0.0137 | 0.0249 | 0.0706 | 0.0192 |

## Layer 3 - constant body-frame twist fit (PUSH trajectories only)

| arm | push completed | aborted before completion | twist fitted | fit pos RMSE | fit angle RMSE | extrapolation pos RMSE | straight-line share |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 28 | 0 | 26 | 0.0048 | 0.0013 | 0.0255 | 0.885 |
| N | 897 | 63 | 882 | 0.0068 | 0.0038 | 0.0213 | 0.728 |
| O | 0 | 0 | 0 | - | - | - | - |
| P | 872 | 88 | 772 | 0.0068 | 0.0029 | 0.0275 | 0.455 |

## D.1 frozen engineering gates (A5-corrected denominators)

| gate | value | threshold | pass |
|---|---:|---:|:--:|
| G2 effective push 2 s (primary: solo_central+dual_same_side) | 0.9250 | 0.90 | yes |
| G2 (secondary: four pushable classes) | 0.9375 | 0.90 | yes |
| G3 false stable | 0.0000 | 0.05 | yes |
| G3 coverage (all 960 executions) | 0.6510 | - | - |
| G4 N/P displacement median (primary) | 0.5317 | 0.80 | no |
| G5 position P90 improvement | 0.6019 | 0.20 | yes |
| G5 angle P90 relative change rad (0.0192 -> 0.0215) | 0.1218 | 0.10 | no |
