"""Public-history safety gate with explicit unknown and real observation actions."""
from dataclasses import asdict
import math
from shapely.geometry import Point, mapping
from shapely.ops import unary_union
from .evidence import EvidenceMap
from .geometry import MotionContract, rectangle_pose, envelopes
from .policy import actions, compiled


class Controller:
    def __init__(self,contract=MotionContract(), *, observe=True):
        self.contract=contract;self.map=EvidenceMap(minimum_static_component_area=contract.minimum_static_component_area)
        self.policy=compiled();self.observe=observe;self.initialized=False;self.waypoints=[];self.waypoint=0
        self.valid=True;self.ledger=dict(boundaries=0,pushes=0,views=0,certified_stops=0,uncertified_emergency_stops=0,envelope_violations=0)
        self.pose_history=None;self.has_pushed=False

    def initialize(self,o,pose):
        if pose is None:return
        for r in o['robots']:
            self.map.add_initial_certificate(Point(r['own']['x'],r['own']['y']).buffer(
                self.contract.robot_radius+self.contract.initial_static_separation),source='declared_initial_separation')
        self.map.add_initial_certificate(pose.inner.buffer(self.contract.initial_static_separation),source='declared_initial_separation')
        # Public geometry determines a viewing route around either end of the box.
        n=pose.normal;t=pose.tangent;c=pose.centre
        goal=o['goal']['xy'];forward=(goal[0]-c[0])*n[0]+(goal[1]-c[1])*n[1]
        if forward<0:n=(-n[0],-n[1])
        r=o['robots'][0]['own'];sign=1 if (r['x']-c[0])*t[0]+(r['y']-c[1])*t[1]>=0 else -1
        for normal,tangent in [(-.55,sign*.90),(.55,sign*.90),(.55,0.),(.55,-sign*.90)]:
            self.waypoints.append((c[0]+normal*n[0]+tangent*t[0],c[1]+normal*n[1]+tangent*t[1]))
        self.initialized=True

    def decide(self,o):
        pose=rectangle_pose(o,error=self.contract.pose_error)
        # Declared initially stationary single target; certified observation
        # sweeps avoid it, so observations can intersect its stationary pose set.
        # This refinement is disabled immediately once a PUSH is selected.
        if pose is not None and not self.has_pushed:
            if self.pose_history is not None:
                pose.outer=pose.outer.intersection(self.pose_history.outer)
                pose.inner=unary_union([pose.inner,self.pose_history.inner])
                if pose.outer.is_empty:pose=None
            if pose is not None:
                pose.centre=(pose.outer.centroid.x,pose.outer.centroid.y)
                self.pose_history=pose
        if not self.initialized:self.initialize(o,pose)
        elif pose is not None and not self.has_pushed:
            self.map.add_initial_certificate(pose.inner.buffer(self.contract.initial_static_separation),source='declared_initial_separation')
        self.map.update(o);self.ledger['boundaries']+=1
        r0=o['robots'][0]['own']
        while self.waypoint<len(self.waypoints)and math.dist((r0['x'],r0['y']),self.waypoints[self.waypoint])<.10:self.waypoint+=1
        values={};details={};commands={};regions={}
        target=o['goal']['target_id'];menu={tuple(p)for p in o['entity_legal_pairs'].get(target,[])};global_menu={tuple(p)for p in o['global_legal_pairs']}
        g=(o['goal']['xy'][0]-pose.centre[0],o['goal']['xy'][1]-pose.centre[1])if pose else (0.,0.)
        norm=max(math.hypot(*g),1e-9);goal_actions=[]
        for name,a in actions().items():
            kind=a['kind'];robot=a['robot'];direction=a['direction'];pair=[44,44]
            if kind=='push':pair[robot]=4*direction+1
            elif kind=='observe':pair[robot]=48+4*direction
            commands[name]=dict(target_id=target if kind=='push'else None,a0=pair[0],a1=pair[1],ticks=a['ticks'])
            legal=tuple(pair)in(menu if kind=='push'else global_menu)
            env=envelopes(o,pose,kind,robot,self.contract,direction,target_stationary=not self.has_pushed);regions[name]=env
            evidence=self.map.query(env['static'])
            dynamic=pose is not None and not env['robots'][0].intersects(env['robots'][1])
            if kind=='observe'and pose is not None:dynamic=dynamic and not env['robots'][robot].intersects(pose.outer.buffer(.015))
            if kind=='push'and pose is not None:dynamic=dynamic and not env['robots'][1-robot].intersects(env['target'])
            # More moving objects require an additional uncertainty envelope provider.
            dynamic=dynamic and not any(e['id']!=target for e in o['entities'])
            score=0.
            if kind=='push':
                align=(g[0]*math.cos(direction*math.pi/4)+g[1]*math.sin(direction*math.pi/4))/norm
                score=100+align if align>.90 else -100.
                if align>.90:goal_actions.append(name)
            elif kind=='observe':
                if robot==0 and self.observe and self.waypoint<len(self.waypoints):
                    waypoint=self.waypoints[self.waypoint];dx=waypoint[0]-r0['x'];dy=waypoint[1]-r0['y'];dist=max(math.hypot(dx,dy),1e-9)
                    align=(dx*math.cos(direction*math.pi/4)+dy*math.sin(direction*math.pi/4))/dist
                    score=10+align if align>.30 else -100.
                else:score=-100.
            observed_speed=max(math.hypot(r['own']['vx'],r['own']['vy'])for r in o['robots'])
            bound_valid=self.valid and pose is not None and observed_speed<=.325
            values.update({f'legal:{name}':legal,f'known:{name}':evidence.status!='unknown',
                f'free:{name}':evidence.certified,f'dynamic:{name}':bool(dynamic),f'bound_valid:{name}':bound_valid,
                f'score:{name}':score})
            details[name]=dict(evidence=asdict(evidence),motion_contract=asdict(self.contract),legal=legal,dynamic=bool(dynamic),clearance_lower_m=None if not evidence.certified else 0.)
        # A viewing robot can also need to clear the target's predicted sweep.
        # Reaching geometric visibility does not justify leaving it in that sweep.
        values['need_observation']=any(details[n]['evidence']['status']=='unknown'or
            (details[n]['legal']and not details[n]['dynamic'])for n in goal_actions)
        values['obstacle_blocks_goal']=bool(goal_actions)and all(details[n]['evidence']['status']=='occupied'for n in goal_actions)
        choice=self.policy.select(values);name=choice['option'];certified=name is not None and choice['executable']
        if not certified:
            # Physical emergency behavior is explicitly outside the certificate.
            name='brake';self.ledger['uncertified_emergency_stops']+=1
        elif name=='brake':self.ledger['certified_stops']+=1
        elif name.startswith('view'):self.ledger['views']+=1
        else:self.ledger['pushes']+=1;self.has_pushed=True
        return dict(choice=choice,action=name,command=commands[name],certified=certified,values=values,
                    selected_evidence=details[name],candidates=details,region=regions[name],pose=pose,
                    waypoint=self.waypoint,unknown_goal=values['need_observation'],blocked_goal=values['obstacle_blocks_goal'])

    def online_refinement(self,decision,hypotheses):
        """Finite rollout evidence never promotes unknown geometry to free.

        A dangerous or failed compatible branch prevents empirical admission.
        A safe collection can refine effects only after geometric obligations hold.
        """
        if not decision['certified']:return dict(allowed=False,reason='geometry_or_motion_evidence_unknown')
        if any(h.get('error')or not h.get('path_safe',False)for h in hypotheses):return dict(allowed=False,reason='compatible_counterexample_or_unknown_branch')
        return dict(allowed=bool(hypotheses),reason='conditional_on_spatial_and_motion_certificate')
