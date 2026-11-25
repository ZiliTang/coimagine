"""Optional public progress-event callback for a CAP module during transport PUSH."""
from .phase_aware_cap import translating
from experiments.planar_tasks.relational_search.contract import validate_action
from experiments.planar_tasks.relational_search.motion import ACTIONS


def push_checkpoints(world, action, propose, *, progress_m, max_decisions):
    """Offer a module a new observation at contact acquisition and forward milestones.

    The callback sees (public observation, running command, remaining command ticks,
    event reason). Declining is side-effect free for the physical option. For a
    joint push each robot's forward progress is measured along its own command.
    Simultaneous milestones produce a single review.
    """
    if progress_m <= 0:
        raise ValueError('progress_m must be positive')
    indices = translating(action)
    axes = {i: ACTIONS[action[f'a{i}']].effect_direction for i in indices}
    previous = {}
    travel = {i: 0.0 for i in indices}

    def review(feedback, left):
        if left <= 0 or world.decision_count + 1 >= max_decisions:
            return None
        contact_event = progress_event = False
        for i in indices:
            robot = feedback['robots'][i]
            execution = robot['execution']
            if (not robot['contacting_object'] or not execution.get('direction_valid')
                    or execution.get('stage') in ('brake','switch_brake')
                    or world.env.active_target != action['target_id']):
                previous.pop(i, None)
                travel[i] = 0.
                continue
            own = robot['own']
            xy = (own['x'], own['y'])
            if i not in previous:
                contact_event = True
            else:
                dx, dy = axes[i]
                travel[i] += (xy[0]-previous[i][0])*dx + (xy[1]-previous[i][1])*dy
                if travel[i] + 1e-9 >= progress_m:
                    travel[i] = max(0.,travel[i]-progress_m)
                    progress_event = True
            previous[i] = xy
        if not (contact_event or progress_event):
            return None
        obs = world.env.public()
        replacement = propose(obs, dict(action), left, 'contact' if contact_event else 'progress')
        if replacement is None:
            return None
        replacement = validate_action(replacement, obs)
        if all(replacement[k] == action[k] for k in ('target_id','a0','a1')):
            return None
        return replacement

    return review
