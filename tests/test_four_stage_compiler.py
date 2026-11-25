import pytest

from experiments.planar_tasks.authoring_v2.four_stage.compiler import Extractor, ExtractionError


def host():
    metrics = dict(x=dict(unit="1", domain=[-10, 10]),
                   ok_a=dict(unit="bool"), ok_b=dict(unit="bool"),
                   ok_stop=dict(unit="bool"))
    return dict(metrics=metrics, options={
        name: dict(requires=["var", "ok_" + name], writes=list(metrics), frame=[])
        for name in ("a", "b", "stop")})


SOURCE = '''
def decide(z):
    candidates = []
    if z["x"] > 0:
        candidates.append(("a", z["x"] + 2))
    else:
        candidates.append(("b", 1))
    candidates.append(("stop", -10))
    return candidates
'''


def values(**kw):
    return dict(x=1., ok_a=True, ok_b=True, ok_stop=True, **kw)


def test_original_function_and_extracted_paths_match():
    compiled = Extractor(SOURCE, host())
    for x in (-2., 0., .1, 5.):
        z = values()
        z["x"] = x
        native = compiled.native_candidates(z)
        emitted = sorted(compiled.candidates(z), key=lambda r: r["emission"])
        assert native == [(r["option"], r["score"]) for r in emitted]
        assert [r["option"] for r in emitted] == (["a", "stop"] if x > 0 else ["b", "stop"])


def test_rejection_tries_negative_scored_candidate():
    p = Extractor(SOURCE, host())
    calls = []
    def admit(a):
        calls.append(a)
        return dict(allowed=a == "stop")
    result = p.select(values(), admit)
    assert calls == ["a", "stop"]
    assert result["selected"] == "stop"
    assert result["checks"][-1]["score"] < 0


def test_future_branch_cannot_borrow_unemitted_safe_alternative():
    source = SOURCE.replace('    candidates.append(("stop", -10))\n', "")
    p = Extractor(source, host())
    z = values()
    outcome = dict(updates=dict(x=[1, 2], ok_a=False, ok_b=True, ok_stop=True), path_safe=True)
    report = p.policy.check({k: v if type(v) is bool else [v, v] for k, v in z.items()},
                            "a", [outcome], quality="model")
    assert report["status"] == "counterexample"
    outcome["updates"]["x"] = [-2, -1]
    assert p.policy.check({k: v if type(v) is bool else [v, v] for k, v in z.items()},
                          "a", [outcome], quality="model")["status"] == "proved"


def test_bounded_literal_loop_and_tie_order():
    source = '''
def decide(z):
    candidates = []
    for name in ("b", "a"):
        candidates.append((name, 1))
    return candidates
'''
    p = Extractor(source, host())
    assert p.native_candidates(values()) == [("b", 1), ("a", 1)]
    assert p.policy.select(values())["option"] == "b"


@pytest.mark.parametrize("statement", [
    'z["x"] = 4',
    'while True: pass',
    'candidates.append(("a", observation["truth"]))',
    'candidates.append(("a", z["missing"]))',
    'candidates.append(("a", __import__("os").getpid()))',
    'candidates.append(("a", 1 / z["x"]))',
    'for i in range(10000): candidates.append(("a", 1))',
    'return candidates',
])
def test_unsupported_or_impure_code_rejected(statement):
    source = "def decide(z):\n    candidates = []\n    " + statement + '\n    candidates.append(("stop", -1))\n    return candidates\n'
    with pytest.raises(ValueError):
        Extractor(source, host())


def test_startup_filter_is_part_of_future_selection():
    p = Extractor(SOURCE, host())
    z = values()
    z["ok_a"] = False
    assert p.policy.select(z)["option"] == "stop"
    assert p.candidates(z)[0]["option"] == "a"
    assert not p.candidates(z)[0]["executable"]


def test_units_and_derived_symbols():
    h = host()
    h["metrics"]["clearance"] = dict(unit="m", domain=[0, 10])
    for option in h["options"].values():
        option["writes"] = list(h["metrics"])
    p = Extractor('''
def decide(z):
    candidates = []
    if z["clearance"] >= Q(0.2, "m") and z["good"]:
        candidates.append(("a", 1))
    candidates.append(("stop", -1))
    return candidates
''', h, derived={"good": ["gt", ["var", "x"], ["lit", 0, "1"]]})
    z = {**values(), "clearance": .3}
    assert p.native_candidates(z) == [("a", 1), ("stop", -1)]
    assert p.policy.select(z)["option"] == "a"


def test_conditional_local_reassignment_cannot_silently_change_native_semantics():
    source = '''
def decide(z):
    candidates = []
    score = 1
    if z["x"] > 0:
        score = 2
    candidates.append(("a", score))
    return candidates
'''
    with pytest.raises(ExtractionError):
        Extractor(source, host())


def test_nested_loop_cannot_overwrite_outer_iterator():
    source = '''
def decide(z):
    candidates = []
    for name in ("a", "b"):
        for name in ("stop",):
            candidates.append((name, 1))
        candidates.append((name, 2))
    return candidates
'''
    with pytest.raises(ExtractionError):
        Extractor(source, host())
