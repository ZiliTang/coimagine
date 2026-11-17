# Typed batch semantics v1

This is the historical `feature_runtime="typed_batch"` protocol. New authors now
default to typed_stream; see STREAM_SEMANTICS.md. Explicit typed_batch sessions
continue using the contract below.
Historical campaigns retain their declared runtime and frozen knowledge. To reproduce
the old Python interface, create an explicitly `cached_fork` or `legacy` campaign.
The source marker `SEMANTIC_API = 'typed-batch-v1'` selects the compiled interface
in standalone Features/rollout and downstream parallel collectors as well.

## Author contract

Submit one `.py` source in Cython pure-Python syntax, at most48KiB. It contains:

```python
import cython
SEMANTIC_API = 'typed-batch-v1'
NAMES = ['my_signal']  # 1..24 named channels in initial author campaigns
def initial_memory() -> dict: ...
def update_memory(memory: dict, event: dict) -> dict: ...
def prepare_context(memory: dict, observation: dict) -> dict:
    return dict(values=[...], proposals=[...])
def evaluate_candidates(context: cython.const[cython.double][:],
                        candidates: cython.const[cython.double][:, :],
                        out: cython.double[:, :]) -> None:
    i: cython.Py_ssize_t
    for i in range(candidates.shape[0]):
        out[i, 0] = ...
```

`initial_memory/update_memory` keep the exact bounded64-history convention. The
host rebuilds that memory once per current history window; it does not silently
switch to unlimited incremental memory. Events contain only projected legal
observations, preceding requested actions and actual executed ticks.

`prepare_context` runs once per window, before candidate queries. Derive the
state/history-dependent signals here, including entity geometry summaries,
recommendations and numeric margins. Return exactly:

- `values`: 1..4096 numeric scalars, serialized as float64; NaN is permitted for
  unknown state, infinity is not. Context layout is your own documented contract.
- `proposals`: at most16 complete legal commands in target_id/a0/a1/ticks format.
  CAP executes the first recommendation; do not make that first item illegal.

The batch kernel receives a read-only context vector, a read-only N×4 float64
candidate matrix, and a writable N×len(NAMES) float64 output initialized to NaN.
Fill the values for each candidate. NaN becomes public `None`; infinity fails.
Declare loop indices and arithmetic locals with Cython numeric types to get
native loops. Avoid Python dictionaries and per-candidate history scans in this
kernel. Python helpers remain available for preparation, not automatically fast.

Candidate columns:

| Column | Meaning |
|---|---|
|0|index into the current observation's `entities` list; -2 means target_id=None|
|1|robot0 action ID|
|2|robot1 action ID|
|3|requested duration cap in ticks (1..40, each0.05s)|

**-1 means unselected in any column**, not an actual chosen action. -3 in the
entity column marks an invalid recommendation for host rejection. All entity
indices are local to the current observation; use its IDs/goal to build relational
context, never treat an index as a fixed task category. Missing semantics stay NaN.
The candidate array includes both proposals and the current prefix representatives;
do not infer preference from row order. Each row's feature values must depend only
on that candidate and the context, not on other candidates or batch size/order.

Four conditional selections remain target→robot0→robot1→duration. Each level is
evaluated in one batch; later levels depend on earlier sampled choices. All legal
actions remain supported. The framework does not restrict choices to proposals.

## Execution and isolation

Both compilation and native-module execution run inside separate bubblewrap
namespaces, with no repository, network or credentials. The extension is never
imported into the physics/optimizer process. The host accepts JSON results only.
Inputs to the numeric kernel are bytes-backed read-only buffers; each scoring
batch executes in a fresh child. No global/class mutation, reflection, IO,
compiler directives, cimports, pointers or unchecked casts are supported.
Imports: `cython`, `array`, and the deterministic standard-library imports listed
in the old guide. Use separate `import cython`/`import array` statements.
Compiler settings retain bounds/initialization checks and disable fast-math.
This is an interface restriction; the process sandbox remains the safety boundary.

The host lowers the three kernel argument declarations through an augmented
`.pxd` file and verifies three native memoryview arguments in generated C. This
is required because Cython3.0 pure-Python const annotations alone can silently
remain Python objects. Authors still submit one Python-style file; they do not
write `.pxd`/C/build scripts. Use the exact argument names `context,candidates,out`.

The host builds a content-addressed artifact once per source/toolchain version,
verifies the binary hash on reuse, and records compilation wall time and toolchain
identity. Compilation failure/timeout rejects the submission; there is no silent
interpreted fallback for a typed source. Native source still needs semantic and
trajectory verification: compilation alone is not evidence of correct behavior.

Do not copy task-specific old sources or engineering migration examples into a
fresh author experiment. `typed_example.py` is a neutral API-only BRAKE example.

## Host setup / reproduction

Current toolchain: Cython3.0.12, system CPython3.10 ABI, `/usr/bin/gcc`, Linux
bubblewrap/prlimit and `/usr/include/python3.10`. Install the pinned compiler into
the isolated toolchain directory (no system Python replacement):

```bash
.venv/bin/python -m pip install --target outputs/toolchains/cython-3.0.12-cp310 \
  --python-version 3.10 --implementation cp --abi cp310 \
  --platform manylinux2014_x86_64 --only-binary=:all: --no-deps Cython==3.0.12
```

Compilation is capped at90wall seconds,60CPU seconds,1GiB address space and32MiB
per output file; execution retains the normal feature sandbox's limits. Build
wall time is part of author development time. Artifact manifests are frozen with
submissions. Changing the compiler/ABI requires a new build and validation.

References: [Cython pure-Python syntax](https://cython.readthedocs.io/en/3.0.x/src/tutorial/pure.html),
[compiler directives](https://cython.readthedocs.io/en/3.0.x/src/userguide/source_files_and_compilation.html).
