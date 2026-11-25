"""Single-opportunity deferral in the frozen trial06 CAP, including its legacy PUSH reviews."""
import atexit
from collections import Counter
from functools import lru_cache
from math import dist, hypot
from pathlib import Path

from experiments.planar_tasks.relational_search.contract import digest, validate_action
from experiments.planar_tasks.semantic_rl.stream_features import StreamProgram
from experiments.planar_tasks.semantic_rl.typed_compiler import compile_source
from tools import cap_open_recovery_round as baseline

prefix = baseline.old.prefix
host = prefix.s
event = prefix.event
ISO = baseline.old.isolated
FROZEN = {
    'P': '8ac09eee19e57b91cbcaf9f76bd2bc05eba61b8e893e3bf4a0c7caca77708b8e',
    'trial06': '0571e1eb86cedc147c6faeccebbf405cce082614a842440cc1898dfc8711fa48',
}
BROKERS = {}


@lru_cache(maxsize=2)
def source(name):
    directory = baseline.OUT / 'sealed' / name
    text = (directory / 'composed.py').read_text()
    if baseline.digest(text) != FROZEN[name]:
        raise ValueError(f'wrong frozen source: {name}')
    return text


def prepare():
    for name in FROZEN:
        baseline.old.author.host.RoomCampaign(baseline.OUT / 'sealed' / name).verify_frozen()
        compile_source(source(name))
    return dict(FROZEN)


def close():
    for broker in BROKERS.values():
        broker.close()
    BROKERS.clear()


def init_worker():
    close()
    BROKERS.update((name, StreamProgram(source(name))) for name in FROZEN)
    atexit.register(close)


def prepare_features(obs, memory, plan, proposed, parent, is_review):
    """All quantities come from both public proposals, public memory, and current public sensors."""
    box = memory['compact']
    center = box[:2]
    axis = plan['axis']
    goal = obs['goal']['xy']
    gap = dist(center, goal)
    toward = [(g - c) / max(gap, 1e-9) for g, c in zip(goal,center)]
    now = obs['tick']
    hist = memory['slot']['samples']
    old = next((sample for sample in reversed(hist) if now - sample[0] >= 140), None)
    recent = [x for x in hist if now-x[0] <= 145]
    pusher = obs['robots'][plan['robot']]
    own = pusher['own']
    ax, ay = box[2:4]
    nx, ny = axis
    half_width = .5*(abs(nx*ax + ny*ay)*box[4]+abs(-nx*ay + ny*ax)*box[5])
    station = (center[0]-(half_width+.34)*nx, center[1]-(half_width+.34)*ny)
    active = obs['robots']
    return {
        'goal_gap_m': gap,
        'recent_goal_advance_m': dist(old[1:3], goal)-gap if old else 0.,
        'stalled_ticks_145': float(sum(x[3] for x in recent)),
        'axis_goal_alignment': nx*toward[0]+ny*toward[1],
        'proposed_travel_m': float(plan['length']),
        'station_distance_m': dist((own['x'],own['y']),station),
        'remaining_seconds': float(own_seconds(obs)),
        'pusher_speed_m_s': hypot(own['vx'],own['vy']),
        'target_contacts': float(sum(bool(r['contacting_object']) for r in active)),
        'static_contacts': float(sum(bool(r['contacting_static']) for r in active)),
        'parent_pushers': float(sum(parent[k]<32 for k in ('a0','a1'))),
        'candidate_pushers': float(sum(proposed[k]<32 for k in ('a0','a1'))),
        'followup': float(bool(plan['followup'])),
        'push_review': float(is_review),
    }


def own_seconds(obs):
    return obs['robots'][0]['remaining_seconds']


class EntryMember(ISO.IsolatedCAP):
    def __init__(self, room, gate, target):
        super().__init__(BROKERS['trial06'])
        self.room = room
        self.gate = gate
        self.target = target
        self.entries = []
        self.matched = False
        self.history = None
        self.committed = None
        self.veto_review = None

    def update(self, history):
        if history.index != self.index+1:
            raise ValueError('missed committed boundary')
        obs_event = host.phase.cap.current_event(history)
        tick = obs_event['observation']['tick']
        dt = None if history.previous_tick is None else tick-history.previous_tick
        row = self.advance(dict(obs_event, executed_ticks=dt))
        self.memory = row['memory']
        self.history = history.sha256
        self.committed = row
        self.index = history.index
        if self.veto_review is not None and self.veto_review <= tick:
            # A vetoed review which switched to a parent action can be followed at the
            # same tick by an actual committed history update. No stale proposal may own it.
            if self.veto_review==tick:
                self.clear_pending(self.memory)
            self.veto_review = None

    @staticmethod
    def clear_pending(memory):
        state = memory['recovery']
        if state.pop('pending',None) is not None:
            state.pop('command',None)

    def choose(self, row, obs, kind, running=None):
        props = row['proposals']
        memory = row['memory']
        state = memory['recovery']
        plan = state.get('pending')
        review = running is not None
        def usable(raw):
            try:
                valid=validate_action(raw,obs)
            except ValueError:
                if review: return None
                raise
            if review and event.same_control(valid,running):
                return None
            return valid
        current=usable(props[0])
        if not plan or state.get('command')!=props[0]:
            return current
        # trial06's prepare_context inserts its proposal at index zero; the mother's
        # original choice is immediately after it, including her earlier elective patch.
        parent=usable(props[1])
        if current==parent:
            return current
        feat=prepare_features(obs,memory,plan,props[0],props[1],review)
        entry=dict(id=len(self.entries),room=self.room,tick=obs['tick'],kind=kind,
                   history=self.history,observation_hash=digest(obs),memory_hash=digest(memory),
                   features=feat,requested=props[0],parent_requested=props[1],running=running)
        veto=False
        if self.target is not None and entry['id']==self.target['id']:
            for k,v in entry.items():
                if v!=self.target[k]:
                    raise ValueError(f"fork prefix mismatch at {self.room} opportunity {entry['id']}: {k}")
            veto = self.matched = True
        elif self.gate:
            if self.gate['kind']=='always_defer':
                veto=True
            elif self.gate['kind']!='zero':
                from tools.cap_entry_residual_fit import should_defer
                veto=bool(should_defer(self.gate,feat))
        entry['deferred']=veto
        self.entries.append(entry)
        if veto:
            if not review:
                self.clear_pending(memory)
            elif parent is not None:
                self.veto_review=obs['tick']
            return parent
        return current

    def propose(self,obs):
        return self.choose(self.committed,obs,'boundary')


def rollout(room,fixture,gate=None,defer_at=None,parent=False):
    """Take by default; defer_at is an exact entry from a natural (zero) rollout."""
    if defer_at is not None and gate:
        raise ValueError('single fork or learned gate, not both')
    if parent:
        member = ISO.IsolatedCAP(BROKERS['P'])
        _,row=prefix.run_member((room,fixture,False),member,ISO.preview)
        return dict(result=row['result'],entries=[],matched=False)
    member=EntryMember(room,gate,defer_at)
    counts=Counter()
    def decide(world,obs,running,left,reason):
        elapsed=obs['tick']-host.observe(world)['tick']
        row=member.advance(dict(observation=obs,action=running,executed_ticks=elapsed))
        chosen=member.choose(row,obs,reason,running)
        counts['check_'+reason]+=1
        if chosen is not None:
            counts['changed_'+reason]+=1
        return chosen
    with host.phase.world(fixture) as (world,_):
        member.update(world.history)
        review=event.PushReview('progress',decide,progress_m=.20,contact_ticks=3)
        pending=None
        while not host.ended(world):
            obs=host.observe(world)
            action=pending if pending is not None else member.propose(obs)
            pending=host.tick_action(world,action,'split_precision',member.update,counts,review=review)
        outcome=host.output(world,counts)
    if defer_at is not None and not member.matched:
        raise ValueError(f'{room}: frozen target was not reached')
    return dict(result=outcome,entries=member.entries,matched=member.matched)
