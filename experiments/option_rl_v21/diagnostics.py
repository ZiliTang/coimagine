"""Read-only execution labels, excluded from policy observations and rewards."""
import math
from coimagine.random_start import spec as R

FIELDS = ('motor_force_integral_Ns', 'effort_integral_s', 'max_motor_force_N',
          'opposing_contact_s', 'stalled_opposing_contact_s', 'yield_robot_s',
          'yield_active_robot_s', 'yield_blocked_robot_s',
          'wait_object_contact_robot_s', 'yield_object_contact_robot_s', 'nonfinite_force_ticks')
REPORT_FIELDS = FIELDS + ('mean_motor_force_N',)
DEFINITION = dict(force='actual motor force after coordinator gate; not object contact force',
    opposing='both physically contacting; both motor forces >1N; cosine <-0.5',
    stalled='opposing plus object speed <0.02m/s and abs(omega)<0.05rad/s',
    object_truth='offline diagnostics only; never included in observations, masks or rewards',
    effort='integral of mean squared force/Fmax over the two robots; seconds',
    robot_s='sum across robots, may reach twice episode duration')


class ExecutionDiagnostics:
    def __init__(self):
        self.values = dict.fromkeys(FIELDS, 0.)

    def update(self, env, observations, dt):
        forces = env.gate_post
        if not all(math.isfinite(v) for f in forces for v in f):
            self.values['nonfinite_force_ticks'] += 1
            return
        lengths = [math.hypot(*f) for f in forces]
        self.values['motor_force_integral_Ns'] += sum(lengths)*dt
        self.values['effort_integral_s'] += .5*sum((n/R.MAX_FORCE)**2 for n in lengths)*dt
        self.values['max_motor_force_N'] = max(self.values['max_motor_force_N'], *lengths)
        contacts = [bool(o.contacting_object) for o in observations]
        if all(contacts) and min(lengths) > 1.:
            cosine = sum(a*b for a, b in zip(forces[0], forces[1]))/(lengths[0]*lengths[1])
            if cosine < -.5:
                self.values['opposing_contact_s'] += dt
                obj = env.object_state()  # offline label, never returned to the actor
                if math.hypot(obj.vx, obj.vy) < .02 and abs(obj.omega) < .05:
                    self.values['stalled_opposing_contact_s'] += dt
        for i, slot in enumerate(env.slots):
            if slot.pending is not None:
                continue
            if slot.action.kind == 'yield':
                debug = slot.last_info.get('yield_debug') or {}
                self.values['yield_robot_s'] += dt
                self.values['yield_active_robot_s'] += dt*bool(debug.get('active'))
                self.values['yield_blocked_robot_s'] += dt*bool(debug.get('blocked'))
                self.values['yield_object_contact_robot_s'] += dt*contacts[i]
            elif slot.action.kind == 'wait':
                self.values['wait_object_contact_robot_s'] += dt*contacts[i]


def combine(diagnostics, seconds):
    result = {key: (max((d.get(key, 0.) for d in diagnostics), default=0.)
                    if key == 'max_motor_force_N' else sum(d.get(key, 0.) for d in diagnostics))
              for key in FIELDS}
    result['mean_motor_force_N'] = result['motor_force_integral_Ns']/max(2*seconds, 1e-9)
    return result
