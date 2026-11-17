# Incremental semantics — typed-stream-v1

New SemanticRLCampaign sessions default to `feature_runtime="typed_stream"`.
Read the supplied `stream_example.py`. Old typed-batch-v1 and legacy campaigns
retain their exact bounded-window behavior and original knowledge seals.

Submit Cython pure-Python source with SEMANTIC_API='typed-stream-v1', literal
unique NAMES (candidate features, 1..64), STATE_NAMES (state features, 1..32), and
optional HISTORY_LENGTH (1..32; default8). Use the same typed evaluate_candidates
signature as typed-batch-v1. Compilation, bounds checking and isolation remain.

- initial_memory() returns a JSON dictionary.
- update_memory(memory,event) runs ONCE per new decision event. Its only inputs
  are explicit memory and the new projected legal observation, preceding requested
  action, and actual executed_ticks (None for initial observation). Memory is at
  most8192 JSON bytes. It persists for the episode, not a rolling raw64 window.
- prepare_context(memory,observation) runs once after the update; return exactly
  {values: numeric context, state: vector ordered by STATE_NAMES, proposals: list}.
  Context: 1..4096 numbers, NaN permitted; state: finite numeric/None values.
  At most16 full proposals; CAP executes the first legal proposal, as before.
- evaluate_candidates(context,candidates,out) computes NAMES for each candidate;
  the native readonly memoryview signature and -1/-2/-3 encoding are unchanged.

Put candidate-independent descriptive semantics in state. Use explicit memory for
history-dependent signals and persistent observed-entity ID bindings. Put candidate
compatibility/scale features in evaluate_candidates. No true-state inputs. Entity
IDs are opaque stable labels; local matrix indices are only current-packet indices.
Unknown values must be None/NaN, not invented geometry. A disappearing object need
not be forgotten if your bounded explicit memory retains its observed evidence.

The host stores the last H state vectors in oldest-to-newest order. Each has32
reserved scalar slots, a valid flag and elapsed age in seconds. Left padding has
unknown values, valid0 and unknown age; unused reserved slots of valid frames are0.
The scorer input is: candidate semantic slots (64 default), H*(32+2) state-history
coordinates, recommendation flag, prefix flags and action encoding. Source SHA,
named candidate slots and semantic_protocol bind the representation. Every PPO
trace stores the actual feature panel; optimization does NOT rerun author code.
The geometric residual uses compact entity centroids/counts/timestamps from these
H boundaries and the current legal robot/goal packet, not the full raw history.
Its existing residual regularizer is unchanged. This is a new representation, not
an assertion that the summaries retain all information needed for every task.

The stream cursor keeps one immutable encoded raw event and a versioned causal
hash chain. Full raw public trajectories remain in rollout recordings for audit
and offline feature re-extraction after semantic revisions. Online consumption
must start at episode start or a matching snapshot; skipped/divergent events fail
explicitly. Repeating a boundary does not update memory. New episodes reset it.
Use Features.new_history(); do not pass a truncated legacy History as full memory.

Explicit state is returned through JSON, never authored pickle. The native module
loads in a sandbox; each update/score executes in a fresh fork from a pristine
module. No author globals persist across operations. Broker restart preserves
host-held explicit memory. Features.snapshot()/restore(snapshot,history) and
StreamHistory.snapshot()/restore() jointly resume the same semantic version and
cursor; they do not checkpoint the physical environment. A source/protocol change
requires replay from original raw data and refitting/recollecting as appropriate;
old vectors/PPO data cannot be silently reused. Reserved slots keep dimensions
stable as named signals are added, but do not waive re-extraction/refit/version
checks; new state signals are appended, never silently reinterpreted in old data. Old raw rollout hashes are checked
under their original protocol when re-extracting a new source.

Budget accounting: CAP one advance request per new boundary; conditional actor
one advance plus four scoring batches; bootstrap one advance plus one root batch.
These count as5/2 feature requests respectively, not4/1. Repeated scoring and
failed requests are charged. Offline source recompilation remains development cost.
The default example is a neutral BRAKE policy, not a solved task or learned expert.

## Frozen-source training backend

Author sessions remain isolated. For a frozen, reviewed source with explicit
memory and no mutable module-global dependence, collectors may explicitly use:

```python
Features(source, execution_backend='trusted_inprocess',
         trusted_source_sha256=digest(source))
```

This loads native code into the collector and is not a security sandbox. It keeps
legal-input projection, readonly numeric buffers, validation, explicit memory and
saved-panel accounting. It does not promise per-query fresh module globals or an
in-process timeout. Verify isolated/in-process panel and rollout equality for each
admitted source. The execution backend does not change the semantic protocol or
model inputs. Parallel Config/CLI supports execution_backend/--execution-backend;
the default remains isolated and trusted jobs pin the model's source digest.

The system sandbox uses CPython3.10 while the training interpreter currently uses
CPython3.11. trusted_native recompiles the already generated, bounds-checked C for
the host ABI inside a build sandbox, with a separate content-addressed cache bound
to the source build, headers, compiler and SOABI. Compilation and first loading
must be counted separately from warm sampling. No author API calls are enabled by
selecting this runtime. No changes to implicit recommendation features accompany
this backend comparison.
