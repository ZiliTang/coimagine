"""A3: per-variant end-state accounting, safety checks, and the paired old/new comparison.

    python experiments/option_prediction_mechanism_v1/phase_a/analyse_a3.py \
        --tags w0 w020 w030 --split dev \
        --out outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/a3_dev.json

Every ratio is reported with its denominator. The safety checks come first by contract:

  motion_false_confirmation  the criterion fired while the truth window was NOT quiet;
  stale_false_confirmation   the criterion fired while its evidence lagged the present by more
                             than the frozen freshness, i.e. a stale quiet history was promoted
                             to a present guarantee.

Only after those are clear are the static confirmation rate and the confirmation latency read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.local_contact_option.contract import SURFACE_FRESHNESS  # noqa: E402

ABORT_REASONS = ("contact_lost", "patch_lost", "patch_support_lost", "surface_unobserved",
                 "approach_timeout", "not_ready")


def load(out_root: Path, run_id: str, tag: str, split: str) -> List[dict]:
    path = out_root / run_id / f"executions_{tag}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} (run with --tag {tag})")
    return [json.loads(line) for line in path.open()]


def summarise(records: List[dict]) -> Dict[str, object]:
    total = len(records)
    counts: Counter = Counter()
    by_variant: Dict[str, Counter] = defaultdict(Counter)
    latencies: List[float] = []
    lags: List[float] = []
    spans: List[float] = []
    for record in records:
        metrics = record["metrics"]
        variant = record["variant"]
        primary = metrics.get("primary_exit_reason") or ""
        locally = bool(metrics.get("locally_settled"))
        settled = bool(metrics.get("settled_observed"))
        after_abort = bool(metrics.get("settled_after_abort"))
        exit_reason = metrics["exit_reason"]
        truth_ok = bool(metrics.get("window_truth_low_speed"))
        surface_ok = bool(metrics.get("evidence_truth_surface_low"))
        lag = metrics.get("settle_evidence_lag")
        span = metrics.get("settle_evidence_span")

        bucket = ("normal_completion" if settled else
                  "settled_after_abort" if after_abort else
                  "timeout" if exit_reason == "settle_timeout" else
                  "unobservable" if exit_reason == "settle_unobserved" else
                  "aborted" if primary in ABORT_REASONS else "other")
        # two different safety questions, kept apart:
        #   estimator_false_confirmation -- the local surface truth was still moving
        #   whole_body_not_at_rest       -- the estimator was locally right, but the object was
        #                                   rotating/translating above the global thresholds
        #                                   (the declared local blind spot)
        estimator_false = bool(locally and not surface_ok)
        blind_spot = bool(locally and surface_ok and not truth_ok)
        stale_false = bool(locally and lag is not None and lag > SURFACE_FRESHNESS + 1e-9)

        counts[bucket] += 1
        counts["push_completed"] += int(bool(metrics.get("push_completed")))
        counts["false_stable_truth"] += int(bool(metrics.get("false_stable")))
        counts["locally_settled"] += int(locally)
        counts["estimator_false_confirmation"] += int(estimator_false)
        counts["whole_body_not_at_rest_blind_spot"] += int(blind_spot)
        counts["motion_false_confirmation"] += int(locally and not truth_ok)
        counts["stale_false_confirmation"] += int(stale_false)
        if locally:
            if lag is not None:
                lags.append(float(lag))
            if span is not None:
                spans.append(float(span))
            stage = metrics.get("stage_seconds") or {}
            latencies.append(float(stage.get("brake", 0.0)) + float(stage.get("settle", 0.0)))

        counter = by_variant[variant]
        counter["n"] += 1
        counter[bucket] += 1
        counter["locally_settled"] += int(locally)
        counter["estimator_false_confirmation"] += int(estimator_false)
        counter["whole_body_not_at_rest_blind_spot"] += int(blind_spot)
        counter["stale_false_confirmation"] += int(stale_false)

    def entry(key: str) -> Dict[str, object]:
        return {"count": counts[key], "denominator": total,
                "rate": counts[key] / total if total else None}

    return {
        "executions": total,
        "counts": dict(counts),
        "rates_with_denominator": {
            key: entry(key) for key in
            ("normal_completion", "settled_after_abort", "locally_settled", "timeout",
             "unobservable", "aborted", "push_completed", "estimator_false_confirmation",
             "whole_body_not_at_rest_blind_spot", "motion_false_confirmation",
             "stale_false_confirmation")},
        "estimator_false_confirmation_denominator_is_locally_settled": {
            "count": counts["estimator_false_confirmation"],
            "denominator": counts["locally_settled"],
            "rate": counts["estimator_false_confirmation"] / max(1, counts["locally_settled"]),
        },
        "motion_false_confirmation_denominator_is_locally_settled": {
            "count": counts["motion_false_confirmation"], "denominator": counts["locally_settled"],
            "rate": counts["motion_false_confirmation"] / max(1, counts["locally_settled"]),
        },
        "confirmation_latency_seconds": _percentiles(latencies),
        "evidence_lag_seconds": _percentiles(lags),
        "evidence_span_seconds": _percentiles(spans),
        "by_variant": {variant: dict(counter) for variant, counter in sorted(by_variant.items())},
    }


def _percentiles(values: List[float]) -> Dict[str, Optional[float]]:
    values = sorted(values)
    if not values:
        return {"n": 0, "p50": None, "p90": None, "max": None}

    def q(fraction: float) -> float:
        position = fraction * (len(values) - 1)
        low = int(position)
        high = min(len(values) - 1, low + 1)
        weight = position - low
        return values[low] * (1 - weight) + values[high] * weight

    return {"n": len(values), "p50": q(0.5), "p90": q(0.9), "max": values[-1]}


def paired(old: List[dict], new: List[dict]) -> Dict[str, object]:
    """Per (root, variant) transitions between the two protocols, same initial state."""

    index = {(r["root_id"], r["variant"]): r for r in old}
    transitions: Counter = Counter()
    new_normal: Counter = Counter()
    for record in new:
        key = (record["root_id"], record["variant"])
        previous = index.get(key)
        if previous is None:
            continue
        before = ("normal" if previous["metrics"].get("settled_observed")
                  else "abort" if (previous["metrics"].get("primary_exit_reason")
                                   or previous["metrics"]["exit_reason"] in ABORT_REASONS)
                  else "unsettled")
        metrics = record["metrics"]
        after = ("normal" if metrics.get("settled_observed")
                 else "after_abort" if metrics.get("settled_after_abort")
                 else "abort" if (metrics.get("primary_exit_reason")
                                  or metrics["exit_reason"] in ABORT_REASONS)
                 else "unsettled")
        transitions[f"{before}->{after}"] += 1
        if after == "normal":
            new_normal[record["variant"]] += 1
    return {"transitions": dict(transitions), "n": sum(transitions.values())}


def render(payload: dict) -> str:
    lines = ["# A3: per-variant end-state accounting and selection", ""]
    for tag, block in payload["tags"].items():
        r = block["rates_with_denominator"]
        lines += [f"## tag `{tag}`  (n={block['executions']})", "",
                  "| outcome | count | denominator | rate |", "|---|---:|---:|---:|"]
        for key in ("normal_completion", "settled_after_abort", "locally_settled", "timeout",
                    "unobservable", "aborted", "estimator_false_confirmation",
                    "whole_body_not_at_rest_blind_spot", "motion_false_confirmation",
                    "stale_false_confirmation"):
            row = r[key]
            lines.append(f"| {key} | {row['count']} | {row['denominator']} | "
                         f"{row['rate']:.3f} |")
        est = block["estimator_false_confirmation_denominator_is_locally_settled"]
        mfc = block["motion_false_confirmation_denominator_is_locally_settled"]
        latency = block["confirmation_latency_seconds"]
        lag = block["evidence_lag_seconds"]
        lines += ["",
                  f"- **估计器**误判（真值局部表面速度仍超门槛）：{est['count']}/{est['denominator']}"
                  f" = {est['rate']:.3f}",
                  f"- 整体未静止但局部正常（本地传感盲区）："
                  f"{r['whole_body_not_at_rest_blind_spot']['count']}/{mfc['denominator']}",
                  f"- 两者都算的 motion_false_confirmation（真值整体未静止）：{mfc['count']}"
                  f"/{mfc['denominator']} = {mfc['rate']:.3f}（与上面两项不互斥）",
                  f"- 确认延迟 s：p50={_f(latency['p50'])} p90={_f(latency['p90'])} "
                  f"max={_f(latency['max'])}",
                  f"- 证据滞后 s：p50={_f(lag['p50'])} p90={_f(lag['p90'])} max={_f(lag['max'])}",
                  ""]
    if "paired_vs_old" in payload:
        lines += ["## 与既有回归集的配对转移（旧协议 -> 新协议）", "",
                  f"配对 {payload['paired_vs_old']['n']} 条："
                  f"{payload['paired_vs_old']['transitions']}", ""]
    return "\n".join(lines)


def _f(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="option-prediction-mechanism-v1")
    parser.add_argument("--out-root", default="outputs/option_prediction_mechanism_v1")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--pair-with", default=None, help="tag to use as the 'old' protocol")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_root = ROOT / args.out_root
    payload: Dict[str, object] = {"split": args.split, "tags": {}}
    loaded: Dict[str, List[dict]] = {}
    for tag in args.tags:
        records = load(out_root, args.run_id, tag, args.split)
        loaded[tag] = records
        payload["tags"][tag] = summarise(records)
    if args.pair_with:
        payload["paired_vs_old"] = {"reference_tag": args.pair_with,
                                    **paired(loaded[args.pair_with], loaded[args.tags[-1]])}
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1, sort_keys=True))
    text = render(payload)
    target.with_suffix(".md").write_text(text)
    print(text)
    print("A3_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
