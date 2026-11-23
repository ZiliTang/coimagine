from experiments.planar_tasks.semantic_rl import phase_aware_cap as policy


def observation(tick):
    return dict(tick=tick,robots=[dict(own=dict(x=1.,y=2.)),dict(own=dict(x=2.,y=3.))])


def test_only_repeated_stalled_contact_changes_push_assignment(monkeypatch):
    controller=policy.ContactAwareCAP()
    push=dict(target_id='target',a0=1,a1=44,ticks=30)
    other=dict(push,a0=44,a1=1)
    monkeypatch.setattr(policy,'recovery_options',lambda *_:[(1,other)])
    controller.finish_contact(push,observation(0),observation(20),'contact_timeout',20)
    assert controller.choose_contact(observation(20),push)==push
    controller.finish_contact(push,observation(20),observation(40),'contact_timeout',20)
    assert controller.choose_contact(observation(40),push)==other
    assert controller.choose_contact(observation(160),push)==push


def test_precision_caps_only_near_goal_direct_push(monkeypatch):
    controller=policy.ContactAwareCAP(precision=True)
    monkeypatch.setattr(policy,'fitted',lambda *_:dict(x=0.,y=0.))
    cmd=dict(target_id='target',a0=3,a1=44,ticks=40)
    obs=dict(goal=dict(xy=[.25,0.],target_id='target'),entity_legal_pairs={'target':[[1,44],[3,44]]})
    actual=controller.choose_push(obs,cmd,28)
    assert actual==dict(cmd,a0=1,ticks=12)
    assert cmd==dict(target_id='target',a0=3,a1=44,ticks=40)
    obs['goal']['xy']=[0.,.25]
    assert controller.choose_push(obs,cmd,28)==dict(cmd,ticks=28)
