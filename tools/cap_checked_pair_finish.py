"""Wait for both accepted, on-time nominations, then run a single paired evaluation."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from tools import cap_checked_pair_author as authors
from tools import cap_checked_pair_evaluate as evaluation
from tools.cap_checked_pair_audit import audit_subset


def arm_table():
    root = authors.OUT
    activation = authors.read(root / 'activation.json')
    for filename, expected in activation['tools'].items():
        if authors.sha(authors.ROOT / filename) != expected:
            raise ValueError(f'author host changed: {filename}')
    check = root / 'conformance/result.json'
    if not authors.read(check)['all_match'] or authors.sha(check) != activation['conformance_sha256']:
        raise ValueError('shared starting program failed historical conformance')
    if authors.sha(root / 'protocol.json') != activation['protocol_sha256']:
        raise ValueError('formal authoring protocol changed')
    table = {
        'detour': dict(composition='P绕行.v1@G程序.v1', campaign=str(authors.study.ORIGIN),
                       sha256=evaluation.HISTORICAL['detour']),
        'parent': dict(composition='P保守推进.v1@G程序.v1', campaign=str(authors.HISTORY),
                       sha256=evaluation.HISTORICAL['parent']),
    }
    for public, role, label in [('control', 'control', '常规修订'), ('checker', 'checked', '检查反馈修订')]:
        nomination = authors.read(root / 'frozen' / f'{public}.json')
        clock = authors.read(root / 'ledger' / public / 'clock.json')
        if (nomination['arm'] != public or
                not clock['start'] <= nomination['nomination_mtime'] <= clock['deadline']):
            raise ValueError(f'invalid submission clock for {public}')
        name = nomination['name']
        relative = Path('parent') if name == 'parent' else Path(public) / name
        checked = root / 'submissions' / relative
        campaign = root / 'sealed' / relative
        receipt = authors.check_receipt(checked)
        if (str(checked) != nomination['checked'] or str(campaign) != nomination['campaign'] or
                receipt['source_sha256'] != nomination['source_sha256'] or
                authors.sha(campaign / 'manifest.json') != nomination['campaign_manifest_sha256']):
            raise ValueError(f'wrong checked program for {public}')
        table[role] = dict(composition=f'P{label}.v1@G程序.v1', campaign=str(campaign),
                           submission=str(checked), sha256=nomination['source_sha256'],
                           nomination=name)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    opts = parser.parse_args()
    out = evaluation.DEFAULT_OUT
    try:
        for arm in authors.ARMS:
            root = authors.OUT / 'ledger' / arm
            while not (authors.OUT / 'frozen' / f'{arm}.json').is_file():
                if (root / 'closed.json').is_file():
                    raise RuntimeError(f'no accepted final nomination for {arm}')
                time.sleep(3)
        evaluation.immutable(out / 'arms.json', arm_table())
        subprocess.run([sys.executable, '-m', 'tools.cap_checked_pair_evaluate', 'evaluate',
                        '--arms', str(out / 'arms.json'), '--workers', str(opts.workers)], check=True)
        audit_subset(out)
    except Exception as exc:
        authors.save(out.parent / 'ERROR.json', dict(error=str(exc), at=time.time()))
        raise


if __name__ == '__main__':
    main()
