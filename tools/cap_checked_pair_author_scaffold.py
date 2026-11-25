"""Expose the frozen trial03 PUSH gate as a shared author starting source.

The ordinary path still checks each generated action in the original order. Offline
reviews pass an arbitrary public action through that same gate with a 40-tick cap.
"""
from textwrap import dedent, indent

from tools.cap_safe_nominal_check_adapter import replace_once
from tools import cap_safe_nominal_study as study

PARENT = 'adf79ad691026c2eefb3a516fd717ba129f9b782cdf905f7e343d8569c8cb432'


def scaffold(source):
    if study.host.digest(source) != PARENT:
        raise ValueError('unexpected historical source')
    begin = source.index('  for act in available_pushes(memory,o,scene,proposals):')
    end = source.index('  base = proposals[0]', begin)
    gate = dedent(source[source.index('    ps = pushers(act)', begin):end])
    gate = gate.replace('if not ps:\n  continue','if not ps:\n  return None, "not_push"')
    for reason in ('illegal_push', 'previous_push_stalled', 'no_shared_continuation_in_sampled_envelope'):
        gate = replace_once(gate, f"reason = '{reason}'\n  continue", f"return None, '{reason}'")
    gate = replace_once(gate, 'if reason:\n  continue', 'if reason:\n  return None, reason')
    gate = replace_once(gate, 'timeout_ticks=TIMEOUT_TICKS)', 'timeout_ticks=TIMEOUT_TICKS), None')
    gate = replace_once(gate, 'continuation(o,act,scene,axis,object_limit)',
                        'continuation(o,act,scene,axis,object_limit,peer_models,peer_hits)')
    source = source[:begin] + '''  for act in available_pushes(memory,o,scene,proposals):
    pair = [act['a0'],act['a1']]
    if auditing is not None and pair not in auditing and min(pair)<32 and act['target_id']==tid:
      auditing.append(pair)
    result = review_push(dict(internal_memory=memory,internal_scene=scene,internal_action=act),o,pair)
    if result['verdict']=='pass':
      return result['permit']
    reason = result['reason']
''' + source[end:]
    source = replace_once(source, 'def candidate_packet(memory,o):',
'''POLICY_CHECK_RULES = {
  'version': 1,
  'widths': {'visible_overlap': ['unknown']},
  'peer_models': {'PUSH': [['co_moving','hypothesis']], 'HOLD': [['stationary','hypothesis']],
                  'YIELD': [['stationary','hypothesis']], 'REPOSITION': [['stationary','hypothesis']], 'FOLLOW': [['stationary','hypothesis']],
                  'MOVE': [['stationary','hypothesis']], 'TURN': [['stationary','hypothesis']]}
}
# Private to one sandboxed audit query. Normal deployed calls never enable it.
AUDIT = {}
def assess_push(memory,o,scene,act,peer_models,peer_hits):
  c,faces,points,obstacles,arena = scene
  tick = o['tick']
  tid = o['goal']['target_id']
''' + indent(gate, '  ') + '''
def review_push(memory, observation, pair):
  o = observation
  if 'internal_memory' in memory:
    m, scene, act = memory['internal_memory'], memory['internal_scene'], memory['internal_action']
  else:
    m = memory
    scene = evidence(m,o)
    act = make(o['goal']['target_id'],pair[0],pair[1],40)
  run = m.get('push_run')
  if run and run.get('timed_out'):
    return dict(verdict='reject',reason='object_no_progress_timeout',peer_robots=[])
  peers, hits = {}, {}
  pkt, reason = assess_push(m,o,scene,act,peers,hits)
  predictions = [dict(robot=i, model=model, scope='hypothesis',
                      used_to_veto=bool(pair in AUDIT.get('considered',[]) and not pkt and hits.get(i)))
                 for i,model in peers.items()]
  return dict(verdict='pass' if pkt else 'reject',reason=reason,peer_robots=list(peers),
              peer_predictions=predictions,permit=pkt)

def audit_policy(memory, observation, menu):
  o = observation
  AUDIT.clear()
  AUDIT['widths'] = []
  considered = []
  pkt = candidate_packet(owned(memory),o,considered)
  considered = [p for p in considered if p in menu]
  AUDIT['considered'] = considered
  act = pkt['command']
  selected = [act['a0'],act['a1']] if pkt['decision']=='permit_push' else None
  widths = [dict(pair=[a,b],source='visible_overlap',lower_m=w,upper_m=None,
                 required_m=.42,conclusion='unknown',pruned=True)
            for a,b,w in AUDIT.pop('widths') if [a,b] in menu]
  return dict(version=1,proposed=considered,selected=selected,
              claim='selected' if selected else 'abstain',widths=widths,peer_predictions=[])

def candidate_packet(memory,o,auditing=None):''')
    source = replace_once(source, 'def continuation(o,action,scene,axis,length):',
                          'def continuation(o,action,scene,axis,length,peer_models,peer_hits):')
    source = replace_once(source, '      future_peer = shift(peer,delta,1.) if peer_drives else peer', '''
      future_peer = shift(peer,delta,1.) if peer_drives else peer
      peer_models[1-i] = 'co_moving' if peer_drives else 'stationary'
      if all(.222<station[k]<arena[k]-.222 for k in (0,1)) and mag(sub(station,future_peer))<.325:
        peer_hits[1-i] = True''')
    source = replace_once(source, '      q=joint_points(f0,f1)', '''      q=joint_points(f0,f1)
      if 'widths' in AUDIT and not q and dot(f0[0],f1[0])>.97 and abs(f0[2]-f1[2])<.04:
        item = [a,b,max(0.,min(f0[4],f1[4])-max(f0[3],f1[3])-.08)]
        if item not in AUDIT['widths']:
          AUDIT['widths'].append(item)''')
    source = replace_once(source, 'not continuation(o,action,scene,axis,min(.055,gap-.067))',
                          'not continuation(o,action,scene,axis,min(.055,gap-.067),{},{})')
    return source


def submission_text(readable):
    """Fit the existing 48 KiB carrier by removing redundant whitespace only.

    Keep the human source separately. Refuse any transformation changing its AST,
    and never remove comments containing host-controlled compiler directives.
    """
    import ast
    import io
    import re
    import tokenize

    if len(readable.encode()) <= 49152:
        return readable
    if re.search(r'^\s*#\s*(cython|distutils)\s*:', readable, re.M):
        raise ValueError('compiler directives are host controlled')
    lines = readable.splitlines(keepends=True)
    depth = 0
    starts = {}
    previous = None
    cuts = {}
    for token in tokenize.generate_tokens(io.StringIO(readable).readline):
        if token.type == tokenize.INDENT:
            depth += 1
        elif token.type == tokenize.DEDENT:
            depth -= 1
        elif token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER, tokenize.COMMENT):
            starts.setdefault(token.start[0], depth)
            if (previous and previous.end[0] == token.start[0] and previous.end[1] < token.start[1]
                    and (previous.type == tokenize.OP or token.type == tokenize.OP)):
                cuts.setdefault(token.start[0], []).append((previous.end[1],token.start[1]))
            previous = token
    result = []
    for lineno, line in enumerate(lines, 1):
        for begin, end in reversed(cuts.get(lineno, [])):
            line = line[:begin] + line[end:]
        if lineno in starts:
            line = ' '*starts[lineno] + line.lstrip(' ')
        result.append(line)
    short = ''.join(result)
    if ast.dump(ast.parse(short)) != ast.dump(ast.parse(readable)):
        raise ValueError('whitespace packing changed the Python AST')
    return short
