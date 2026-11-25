"""Observation-based transport contracts, calibration of physical units, and live execution."""
import json
import math
from pathlib import Path

from coimagine.continuous2d.static_obstacles import StaticObstacleMap
from coimagine.random_start.spec import GEOMETRY_PROFILE, ROBOT_RADIUS, REPOSITION_GAP, REPOSITION_SWEEP_MARGIN, SAFETY_GAP
from experiments.planar_tasks.authoring_v2 import contract_planner
from experiments.planar_tasks.relational_search.footprint_gap import FootprintHistory
from experiments.planar_tasks.relational_search.static_map import update_static_map
from experiments.planar_tasks.relational_search.contract import validate_action
from . import joint_face_change as face
from . import push_phases
from .memory_flow import current_event
from .push_takeover import push_checkpoints

CONTRACT_FILE = Path(contract_planner.__file__).with_name('transport_clearance_contract.json')
ROUTES = ('ccw','cw')
DOSES = (.4,.1)
EFFECTS = ('forward','push_sweep','endpoint_loss','full_sweep')


def option_key(command,dose,route):
    return f"{command['target_id']}/{command['a0']}-{command['a1']}/{dose:.2f}/{route}"


def route_width(relative_angle,route):
    """Disc support against the watched plane along the quarter-circle contact-side arc.

    The intended centre follows the selected geometry profile at the orbit gap.
    Historical models used the 5 cm contact pre-alignment envelope. The controller may deviate; this is a design envelope.
    """
    if route not in ROUTES:
        raise ValueError('unknown face route')
    turn=1 if route=='ccw' else -1
    start=math.pi
    offset=(turn*(relative_angle-start))%(2*math.pi)
    dot=1. if offset<=math.pi/2 else max(math.cos(relative_angle-start),
                                         math.cos(relative_angle-start-turn*math.pi/2))
    offset_gap = .05 if GEOMETRY_PROFILE == "legacy36" else REPOSITION_GAP
    support_mm = 1000*(ROBOT_RADIUS*(1+dot)+offset_gap*max(0.,dot))
    if GEOMETRY_PROFILE == "legacy36":
        return math.ceil(1000*(ROBOT_RADIUS*(1+dot)+offset_gap*max(0.,dot)+SAFETY_GAP))
    # The static sweep rejects equality, so the least admissible whole mm is strictly above.
    return math.floor(support_mm + 1000*REPOSITION_SWEEP_MARGIN) + 1


class TransportPerception:
    def __init__(self, target):
        self.target=target
        self.walls=StaticObstacleMap()
        self.box=FootprintHistory()

    def observe(self, obs):
        update_static_map(self.walls,obs)
        self.box.observe(obs,self.target)

    def clearance(self,obs,command):
        if command['target_id']!=self.target:
            raise ValueError('target-specific footprint required')
        points=next((e['points'] for e in obs['entities'] if e['id']==self.target),[])
        _,axis=push_phases.pusher(command)
        hit=self.walls.forward_surface(points,axis,lateral_margin=.8,min_alignment=.25,min_gap=-.04,max_gap=1.2)
        if hit is None:
            return None
        phi=math.atan2(axis[1],axis[0])+hit.normal_angle
        normal=(math.cos(phi),math.sin(phi))
        bounds=self.box.gap_range(hit.normal_gap,normal)
        if bounds is None:
            return None
        return dict(bounds=bounds,angle=hit.normal_angle,normal=normal,
                    offset=sum(x*n for x,n in zip(hit.surface.start,normal)))


def upper_order(values):
    """One-sided development order statistic; fewer than nine rooms gives no bound."""
    rank=math.ceil(.90*(len(values)+1))
    return sorted(values)[rank-1] if rank<=len(values) else None


def empirical_bounds(model,command,dose,route,angle):
    # No clearance or terminal labels enter this lookup. At fixed heading, route and
    # dose the estimate is independent of the available gap (monotone symbolic gate).
    local=[r for r in model['effects'] if r['pair']==[command['a0'],command['a1']]
           and r['dose']==dose and r['route']==route
           and abs(math.remainder(r['angle']-angle,2*math.pi))<=math.radians(25)]
    bounds={k:[None,None] for k in EFFECTS}
    for k in EFFECTS:
        sign=-1 if k=='forward' else 1
        value=upper_order([sign*r[k] for r in local])
        if value is not None:
            bounds[k]=[math.floor(-1000*value),None] if k=='forward' else [0,math.ceil(1000*max(0.,value))]
    return bounds,len(local)


class TransportPlanner:
    def __init__(self,model,contract=None):
        if model.get('geometry_profile', 'legacy36') != GEOMETRY_PROFILE:
            raise ValueError('transport effect model was calibrated for a different robot geometry')
        self.model=model
        self.contract=json.loads(CONTRACT_FILE.read_text()) if contract is None else contract

    def check(self,command,dose,route,geometry,ready):
        interval=[None,None]
        effects={k:[None,None] for k in EFFECTS}
        count=0
        width=None
        if geometry is not None:
            lo,hi=geometry['bounds']
            interval=[math.floor(1000*(lo-self.model['gap_margin'])),math.ceil(1000*hi)]
            effects,count=empirical_bounds(self.model,command,dose,route,geometry['angle'])
            width=route_width(geometry['angle'],route)
        cell=dict(contact_ready=[int(ready)]*2,clearance=interval)
        # Without a wall observation, the unknown sweep/gap independently prevents admission.
        route_req={self.contract['route_width_metric']:[width if width is not None else 1,None]}
        check=contract_planner.check_candidate(self.contract,
            (f"push-{command['a0']}-{command['a1']}-{dose}",route),cell,effects,route_req,allow_empirical=True)
        check.update(support_rooms=count,route_width_mm=width)
        return check

    def choose(self,perception,obs,ready_commands):
        checks={}
        for dose in DOSES:
            for command in ready_commands:
                geometry=perception.clearance(obs,command)
                for route in ROUTES:
                    checks[option_key(command,dose,route)]=self.check(command,dose,route,geometry,True)
        return contract_planner.first_admitted(checks),checks


def run_push(world,command,dose):
    request=dict(command,ticks=40)
    def review(obs,running,left,reason):
        if reason=='progress':
            return dict(target_id=None,a0=44,a1=44,ticks=1)
    callback=push_checkpoints(world,request,review,progress_m=dose,max_decisions=300)
    return push_phases.push_from_contact(world,request,40,review=callback)


def execute_selected(world,perception,planner,obs,commands):
    """Normal decision entry: check offered ready proposals, dispatch one pair or hold."""
    ready=[]
    rejected=[]
    for c in commands:
        robot,axis=push_phases.pusher(c)
        try:
            validate_action(c, obs)
        except ValueError as exc:
            rejected.append(dict(command=c, reason=str(exc)))
            continue
        status=push_phases.contact_state(world,c,robot,axis)
        if status=='ready':
            ready.append(c)
        else:
            rejected.append(dict(command=c, reason=status))
    selected,checks=planner.choose(perception,obs,ready)
    result=dict(selected=selected,checks=checks,rejected=rejected)
    if selected is None:
        return result
    command,dose,route=next((c,d,r) for c in ready for d in DOSES for r in ROUTES if option_key(c,d,r)==selected)
    geometry=perception.clearance(obs,command)
    phase=run_push(world,command,dose)
    result.update(push_ticks=phase.ticks,push_end_tick=world.env.total_periods,push_status=phase.status)
    if not phase.ticks or world.env.termination:
        return result
    end=current_event(world.history)['observation']
    perception.observe(end)
    points=next((e['points'] for e in end['entities'] if e['id']==command['target_id']),[])
    normal=geometry['normal']
    clearance=None
    if points:
        visible=geometry['offset']-max(x*normal[0]+y*normal[1] for x,y in points)
        bounds=perception.box.gap_range(visible,normal)
        if bounds is not None:
            clearance=math.floor(1000*(bounds[0]-planner.model['gap_margin']))
    result['observed_entry_lower_mm']=clearance
    result['continuation_width_passed'] = clearance is not None and clearance >= checks[selected]['route_width_mm']
    return result
