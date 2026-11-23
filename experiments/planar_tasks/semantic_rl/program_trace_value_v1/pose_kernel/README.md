# Frozen CAP planning kernels

Trusted single-CAP experiments and the seven-CAP trace library use the native pose search by default for frozen CAP009 (SHA256 `48fc2ebcf6f9540f1c457a49e47e9ed7cc0cac3bb9a6a7aa17c2d083e4f5284d`). The first matching worker builds the repository-owned C kernel with GCC. Other workers reuse the hash-checked binary under `outputs/toolchains/cap009-pose/`. To compare against the original implementation, start the experiment with `COIMAGINE_CAP009_POSE=reference`. The paired benchmark below selects its own reference and native arms explicitly.

The C loop constructs 16 obstacle grids from the same publicly observed points, searches eight orientations using the original 12,000-pop limit, and returns the parent path and heap trace statistics. Python computes the original trigonometric directions and per-cell `math.hypot` priorities. Strict heap comparisons, neighbor order, fallback to the best reached node and the original return logic preserve the decision contract. The input is a validated, finite public CAP packet. Grid dimensions outside the prototype range of 1 through 128 per axis use the frozen planner.

From the repository root, using the Python 3.11 environment containing the experiment dependencies:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 COIMAGINE_LIDAR_KERNEL=numpy
outputs/transfers/venv_py311/bin/python -m tools.cap_pose_kernel_benchmark --out outputs/semantic_rl/cap009_pose_reproduction
outputs/transfers/venv_py311/bin/python -m tools.cap_pose_kernel_benchmark --out outputs/semantic_rl/cap009_pose_reproduction --pool-only
```

The first command requires a new output directory, the two transferred libraries and the archived November 19 retention panel/checkpoint. It checks route paths, memory, full context, recorded actions and scalar trace bytes on real prefixes, then executes paired single-worker runs. The second command uses the first run's binary for 12-worker batches. All measurements include Python-to-C input packing. Episode wall times also include the other CAPs when applicable, environment steps and complete replay writes.

## Shared clearance and navigation

The same trusted experiment loaders enable two additional kernels by default for the seven pinned CAP source hashes. Five CAPs use `clearances` to compute eight pushing and sixteen robot movement clearances. All seven use `navigate` to select a destination and search an eight-connected occupancy grid. Each CAP supplies its own points, peer and geometry. The native call preserves the original grid dilation, goal tie-breaking and heap order. The policy still performs the final waypoint smoothing, movement selection and fallback. The grid limit is 128 cells per axis, with the original planner used for larger grids.

Set `COIMAGINE_CAP_SHARED=reference` before starting a worker to use both original helpers, or set it to `clearances` or `navigate` to enable one kernel. The default value is `native`. The CAP009 pose switch is independent.

The shared benchmark replays archived seven-CAP prefixes through both implementations and then measures matched 12-worker batches with all four switch combinations. It uses the same transfer data, frozen panel and checkpoint as the pose benchmark. Run from the repository root with the thread and lidar environment variables above and a fresh output directory.

```bash
PYTHONPATH=. outputs/transfers/venv_py311/bin/python -u -m tools.cap_shared_kernel_benchmark --out outputs/semantic_rl/cap_shared_reproduction
```
