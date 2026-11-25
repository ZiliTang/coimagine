# Authoring a PUSH with a usable continuation

The runnable contract lives in [transport_clearance_contract.json](transport_clearance_contract.json). It names measurement units, starting conditions, promised physical effects, state updates and successor requirements. [contract_planner.py](contract_planner.py) compiles each offered action and continuation using the existing interval checker. The planar runtime binds these names to public measurements, physical predictions and executable options. [Transport runtime](../semantic_rl/guarded_transport.py)

## What to declare

| Name | Meaning |
|---|---|
| `clearance` | Estimated distance from the complete box envelope to the watched static plane along that plane's normal, in mm. The measured lower endpoint includes a development-calibrated allowance for footprint error. |
| `forward` | Signed box-centre progress along the commanded direction during PUSH. The contract requires at least 20 mm. Command dose measures robot progress. |
| `push_sweep` | Largest loss of box-to-plane clearance from the initial boundary through PUSH. Includes rotation of the box. |
| `endpoint_loss` | Loss of clearance at the moment PUSH ends. Extra clearance gained by moving away is conservatively credited as zero loss. |
| `full_sweep` | Largest loss of clearance from the initial boundary through the end of the specified face acquisition, including drift after pushing. |

The three losses have nonnegative lower bounds. For each physical outcome the checker subtracts each loss interval from the initial clearance interval. The minimum modelled gap along both execution windows must be at least 1 mm. The predicted handoff gap must accommodate the chosen face-change route. Candidates with missing geometry or missing dynamics support are held. A counterexample is a violation permitted by the model interval; a conditional admission relies on empirical bounds. Neither status establishes an unconditional physical guarantee.

The route-specific requirement comes from the robot radius, the option's 5 cm pre-alignment distance, the control clearance margin and the quarter-turn arc to the new contact face. The implementation projects a robot disc along that arc onto the watched plane. If the candidate pushes in another direction, the wall angle and the arc change with it. This calculation describes a nominal route envelope. It checks one observed front plane per direction. Other obstacles, a straying controller and robot-to-robot interference can still cause failure.

## Decision and execution

Create one perception state per target and room. Feed it the initial public observation and subsequent observed boundaries. Supply the calibrated physical model and the CAP's offered single-pusher commands to the runtime.

```python
memory = TransportPerception(target_id)
planner = TransportPlanner(physical_model)
memory.observe(public_observation)
result = execute_selected(world=executor, perception=memory, planner=planner,
                          obs=public_observation, commands=proposed_commands)
```

The execution adapter checks current menu legality and tactile face readiness before selection. For each ready heading it checks both declared progress doses and both clockwise and counterclockwise face changes. The preference is long then short, caller's command order, then counterclockwise then clockwise. The PUSH from the first admissible pair is executed. If none qualifies, the caller receives an empty selection. After PUSH the adapter refreshes the public box estimate against the same wall plane and reports the handoff width. Control returns to the policy at the PUSH endpoint. The route is a checked future possibility; starting REPOSITION requires a separate request. The returned record includes all candidates, required widths, empirical support sizes, propagated intervals, individual failed obligations, chosen action, PUSH status and the observed width check. An inadmissible visible face is reported as `push_no_legal_face` under the current 45-degree contract. A subsequent REPOSITION runs only its requested orbit window. See the [public option guide](../relational_search/option_guide.md#translating-push-and-reposition).

## What data can change

Use development runs to calibrate physical quantities. For each eligible development root, measure the actual change in box extent against the plane at each control tick; compare with a run with the added wall removed, if the same command can start there. The latter reduces the risk of using a collision-stopped trajectory as an upper bound on unconstrained motion. The wall-removed preparation can differ. Its displacement is measured from its own contact boundary.

Group samples by joint command, robot progress dose, face-change route and observed relative wall angle. Use one value per room and group, taking the larger of the two swept extents and the smaller forward progress. Use the one-sided 90% development order statistic for each marginal effect, requiring at least nine contributing rooms; the score never reads outcome success. Marginal empirical limits are not a joint 90% certificate. At fixed action, route, angle and physical model, reducing available clearance cannot make an inadmissible pair become admissible.

The historical 35-degree comparison explicitly composes PUSH, orbit and new-face contact in the benchmark driver. Its measurements describe that combination. Run the frozen comparison with `.venv/bin/python -m tools.cap_symbolic_transport` and generate the paired report with `.venv/bin/python -m tools.cap_symbolic_transport_report`. The study records forced outcomes separately from the single live choice. It also compares the symbolic gate and the historical voting gate on exactly the original two doses, heading and counterclockwise route.
