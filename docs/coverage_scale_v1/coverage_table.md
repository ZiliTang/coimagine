# coverage_scale_v1 - data coverage

recipe `coverage-scale-recipe-v1`, at most 32 real windows per root.

## train: 4656 collection roots, 139465 windows, 30.0 windows/root

| family/mode | roots | windows | contact windows | root success | failure windows | mean attempts | acceptance/attempt |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0/broad | 1164 | 34422 | 23995 | 512 | 0 | 6.3 | 15.98% |
| S0/goal | 582 | 16330 | 11130 | 340 | 0 | 5.4 | 18.45% |
| S0/wall | 582 | 17826 | 12666 | 184 | 0 | 76.5 | 1.31% |
| S1/broad | 1164 | 35877 | 24905 | 258 | 0 | 21.2 | 4.71% |
| S1/goal | 582 | 17201 | 11527 | 188 | 0 | 25.2 | 3.96% |
| S1/wall | 582 | 17809 | 12061 | 143 | 0 | 33.5 | 2.99% |

## val: 128 collection roots, 3852 windows, 30.1 windows/root

| family/mode | roots | windows | contact windows | root success | failure windows | mean attempts | acceptance/attempt |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0/broad | 32 | 939 | 662 | 16 | 0 | 6.2 | 16.08% |
| S0/goal | 16 | 454 | 283 | 8 | 0 | 5.8 | 17.39% |
| S0/wall | 16 | 483 | 361 | 4 | 0 | 74.8 | 1.34% |
| S1/broad | 32 | 980 | 636 | 10 | 0 | 14.1 | 7.08% |
| S1/goal | 16 | 509 | 358 | 1 | 0 | 26.0 | 3.85% |
| S1/wall | 16 | 487 | 297 | 3 | 0 | 39.1 | 2.56% |

## test: 128 collection roots, 3760 windows, 29.4 windows/root

| family/mode | roots | windows | contact windows | root success | failure windows | mean attempts | acceptance/attempt |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0/broad | 32 | 949 | 637 | 14 | 0 | 7.6 | 13.22% |
| S0/goal | 16 | 437 | 298 | 8 | 0 | 6.7 | 14.95% |
| S0/wall | 16 | 478 | 339 | 4 | 0 | 83.8 | 1.19% |
| S1/broad | 32 | 968 | 638 | 4 | 0 | 27.7 | 3.61% |
| S1/goal | 16 | 481 | 300 | 5 | 0 | 15.8 | 6.35% |
| S1/wall | 16 | 447 | 292 | 5 | 0 | 22.2 | 4.51% |

## narrow: 768 collection roots, 23071 windows, 30.0 windows/root

| family/mode | roots | windows | contact windows | root success | failure windows | mean attempts | acceptance/attempt |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0/fixed | 384 | 10835 | 7637 | 253 | 0 | 1.0 | 100.00% |
| S1/fixed | 384 | 12236 | 9616 | 41 | 0 | 1.0 | 100.00% |

## success / contact / motion counts over the trained tiers

| split | windows | independent roots | roots with contact | roots with success | translation windows | rotation windows | stationary | contact | success | physical failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 128006 | 4591 | 4591 | 1604 | 75964 | 72312 | 49836 | 89329 | 2275 | 0 |
| val | 3584 | 128 | 128 | 42 | 2022 | 1924 | 1495 | 2446 | 64 | 0 |
| test | 3497 | 128 | 128 | 40 | 2022 | 1922 | 1409 | 2364 | 53 | 0 |
| narrow | 16019 | 626 | 626 | 236 | 10334 | 9732 | 5398 | 11925 | 314 | 0 |

