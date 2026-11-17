# Current LLM authoring entry: shared Option knowledge

New experiments use this entry for **all backends** (including human-mediated GPT
relay and provider APIs). `option_guide.md` is the common public skill contract;
its content is identical for free CAP and Option-search-v3. Historical frozen
baseline/v1/v2/v3 modules remain reproduction entries and are not patched.

```python
from experiments.planar_tasks.authoring import create_session, open_session, drive, verify_frozen

# config is the normal explicit model/budget/development configuration.
# The search condition requires prediction={"version":"nominal-v1","runtime":True}.
s = create_session(new_path, config)  # default method="option_search_v3"
# Or: s = create_session(new_path, config, method="free")
try:
    result = drive(s, complete)       # caller supplies the explicit provider transport
finally:
    s.close()
verify_frozen(new_path)
```

Creating a session does not call an LLM or step physics. GPT relay uses the same
session tools directly: read `instructions.txt`, `option_guide.md`, `contract.json`,
`campaign.json` and, when supplied, `heuristic_api.md` / `prediction_api.md` before
authoring. Relay receipt is evidenced by tool reads; we cannot guarantee human reading
or model understanding. Reopen through `open_session` to retain knowledge checks.

For the automatic tool loop, the entire guide is included in the system message from
the first request. It is added BEFORE input-byte budgeting and event logging, not by
an unaccounted transport wrapper. No model/provider name changes this material.
The guide is also installed as a workspace file and referenced with SHA256 from
`contract.json`. The codebook/physical inputs/runtime executor are unchanged.

`public_knowledge.json` records the host-owned profile, guide/instructions/contract
hashes and knowledge release. Tools reject direct guide edits; on subsequent session
checks, script-side alterations to protected materials fail rather than silently
changing the declared input. Submission records the knowledge profile in its frozen
manifest; `verify_frozen` checks it as well as the existing executor/package seal.
Use this verifier before invoking existing free/search evaluation functions. No new
evaluation, token budget, physics budget or task-specific solution is supplied here.

The pack is independently frozen by `knowledge.freeze()` and checked by
`knowledge.verify()`, including the documented controller source hashes. It adds a
new author-information condition, not a new physical controller. Comparisons against
historical programs must disclose the changed author information. Fair future model
comparisons should use this same pack and declared budgets for every model.

The full prose is in English for one canonical cross-backend input. It covers all
exposed action families, exact continuation/retry semantics, the current translation
implementation, feedback limitations, and nominal predictor cold starts. This is
static public implementation knowledge, not an empirical dynamics model. No API key,
scene truth, benchmark answer or historical performance result is in the guide.
