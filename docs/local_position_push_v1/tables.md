# local_position_push_v1 — test (n=1920)

## 1. completion and contact

| arm | eff push | eff push (primary) | contact ratio | contact_lost | patch_lost | support_lost | approach s | push s | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V | 0.885 | 0.900 | 1.000 | 0.081 | 0.033 | 0.000 | 1.05 | 2.00 | 3.90 |
| X | 0.803 | 0.800 | 1.000 | 0.153 | 0.032 | 0.013 | 1.05 | 2.00 | 3.75 |

## 2. motion (primary denominator, baseline)

| arm | displacement med | p90 | rotation med | push-only med | moved>=2cm | turned>=3deg | no motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| V | 0.265 | 0.354 | 0.059 | 0.199 | 1.000 | 0.525 | 0.000 |
| X | 0.209 | 0.344 | 0.053 | 0.174 | 1.000 | 0.500 | 0.000 |

## 3. force (PUSH periods, baseline)

| arm | applied peak | applied P95 | requested P95 | saturation | normal P95 | tangential P95 | gap P05 | gap med | error P95 | pen max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V | 6.59 | 5.92 | 0.00 | 0.000 | 5.89 | 0.865 | - | - | 0.0000 | 0.00100 |
| X | 6.51 | 5.81 | 5.81 | 0.000 | 5.71 | 0.837 | -0.0005 | -0.0001 | 0.0650 | 0.00079 |

## 4. terminal state and settle

| arm | settled | false stable | after abort | settle timeout | object v med | object w med | robot v max med |
|---|---:|---:|---:|---:|---:|---:|---:|
| V | 0.674 | 0.000 (0) | 0 | 0.198 | 0.0069 | 0.0031 | 0.0000 |
| X | 0.617 | 0.000 (0) | 0 | 0.177 | 0.0070 | 0.0031 | 0.0000 |

## 5. same-root perturbation sensitivity (X - V median, root bootstrap)

| variant | position delta | 95% CI | angle delta (rad) | 95% CI | X pos P90 | V pos P90 | X ang P90 | V ang P90 |
|---|---:|---|---:|---|---:|---:|---:|---:|
| pos_2mm | 0.0002 | [0.0001, 0.0004] | -0.0000 | [-0.0001, 0.0000] | 0.0062 | 0.0037 | 0.0058 | 0.0053 |
| pos_10mm | 0.0001 | [-0.0000, 0.0003] | -0.0002 | [-0.0003, -0.0001] | 0.0099 | 0.0081 | 0.0234 | 0.0209 |
| vel_0.02 | -0.0000 | [-0.0001, 0.0001] | -0.0001 | [-0.0001, -0.0000] | 0.0111 | 0.0106 | 0.0147 | 0.0117 |
| vel_0.05 | 0.0000 | [-0.0001, 0.0001] | -0.0003 | [-0.0005, -0.0002] | 0.0228 | 0.0228 | 0.0278 | 0.0249 |
| noise | 0.0000 | [-0.0001, 0.0001] | -0.0000 | [-0.0001, 0.0001] | 0.0066 | 0.0059 | 0.0115 | 0.0101 |
| delay | 0.0004 | [0.0001, 0.0008] | 0.0000 | [-0.0002, 0.0002] | 0.1471 | 0.1537 | 0.0773 | 0.0670 |
| param | 0.0000 | [-0.0000, 0.0000] | 0.0000 | [-0.0000, 0.0000] | 0.0081 | 0.0090 | 0.0218 | 0.0152 |

## 6. contact combinations and tangential slip (baseline)

| arm | slip median | slip P90 | contact states |
|---|---:|---:|---|
| V | 0.1789 | 0.3201 | {'10': 1997, '11': 2549, '01': 9} |
| X | 0.1715 | 0.3061 | {'10': 2010, '11': 2214, '01': 19, '00': 4} |
