# Duration-only adaptation

`parameter_policy.DurationPolicy.from_supervisor(policy, radius=4)` copies a supervised
Policy into a trainable duration scorer and a separate immutable structural supervisor.
At every legal decision history, the supervisor greedily selects target_id/a0/a1 and
an anchor duration. Only ticks within the inclusive anchor±radius interval and1..40
may be sampled. The selected IDs fix speed as well as family/direction/sign.

This restricts the algorithm's action support; it is not action-space-equivalent to
free CAP. Free CAP and the existing full-action Policy are unchanged. A frozen
supervisor can still select different future structures after durations change the
history. This is not a guarantee of preserving every trajectory or success.

Use `duration_ppo_update` for physical-time GAE/PPO with separately clipped actor and
critic gradients. First three conditional panels have singleton support and contribute
zero policy log probability. The duration panel retains its complete bounded support.
Both raw residual variance and conditional entropy retain the four-panel accounting
(the three structural panels contribute zero variance/entropy).

Use `DurationPolicy.save/load`. Its versioned file embeds the supervisor and checks its
state hash. Ordinary `Policy.load` rejects the different format rather than silently
re-enabling structural exploration. Loaders in existing full-action parallel jobs are
therefore not interchangeable with this experimental collector.

The first bounded host is
`outputs/semantic_rl_authors/local_cap_loop_20261116_01/train_parameters.py`, reached
only after authoring_v2 SemanticRLCampaign supervised recovery and source/backend
verification. Source, model, training subset, timing protocol and stopping criteria
are frozen per campaign. No API or training starts on import.

Tests: `pytest -q tests/test_semantic_rl_parameter_policy.py tests/test_semantic_rl_scale.py`.
