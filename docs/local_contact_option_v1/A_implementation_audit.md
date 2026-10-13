# A.1 implementation audit (read from the code, not from a report)

`continuous2d-skill-v1`; all rows match the task description: **True**

| item | value | source |
|---|---|---|
| window duration | 40 control steps x 0.050 s = 2.00 s | `continuous2d/config.py::PhysicsConfig.window_steps/control_dt; continuous2d/environment.py::execute_joint_skill` |
| control period / physics step | control_dt=0.0500 s (20 Hz), physics_dt=0.00833 s, substeps=6 | `config.py::PhysicsConfig.substeps; backend.py::apply_and_step` |
| force limit | max_force=15.0 N, clamped in backend.clamp_force and skills._clamp_force | `backend.py::clamp_force; skills.py::_clamp_force` |
| velocity inheritance | execute_joint_skill resets controller_state only; body velocities are NOT reset at a window boundary, so a new window inherits them | `environment.py::execute_joint_skill (lines resetting controller_state)` |
| old push: contact test | gap = dot(contact_world - robot_position, direction) - robot_radius; approach switches to push when gap <= 0.005 m | `skills.py::_push_contact` |
| old push: approach stage | PD to the desired point contact - direction*radius, routed by A* (geometry.plan_path) when further than 0.25 m, force clamped to min(max_force, skill.force_limit) | `skills.py::_push_contact approach branch; geometry.py::plan_path` |
| old push: push stage | forward = clamp(push_k_distance*max(0,gap) + push_k_velocity*(push_speed - relative_along), 0, force_limit); lateral PD with push_lateral_kp/kd about the desired contact point | `skills.py::_push_contact push branch` |
| old push: reference frame | push_direction_world is a constant WORLD direction; contact_local is an OBJECT-frame coordinate re-projected through the current object pose every control period (so the skill follows a rotating object and needs object truth) | `skills.py::contact_world, _push_contact` |
| old push: completion / braking | no braking: the window simply ends after window_steps. 'hold' is damped braking F=-hold_kd*v with no position hold | `skills.py::_hold; config.py::ControlConfig.hold_semantics='damped_brake'` |
| task success | success needs the object inside the goal (margin 0.02 m) and below 0.03 m/s / 0.08 rad/s for 0.5 s; the controller never reads it | `environment.py::object_in_goal/object_slow/task_success` |
| controller feedback | skills.control receives the full Observation, which contains the object's global pose and velocity; the old push therefore uses object truth | `skills.py::Observation, control` |
| round trip check | hold window: object pose moved by 0.000e+00 (no self-motion) | `this audit run` |
