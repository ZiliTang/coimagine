import gzip
import json
from tools.cap_trace_value_transfer import audit


def job(path):
    return dict(case_id='case', cap_id='CAP-RR-009', source=str(path),
                summary={'success': False}, fixture_sha256='unknown')


def test_missing_is_not_complete_failure(tmp_path):
    r=audit(job(tmp_path/'missing.gz'))
    assert r['completion']=='missing'


def test_corrupt_gzip_is_not_complete_failure(tmp_path):
    p=tmp_path/'broken.gz';p.write_bytes(b'not gzip')
    r=audit(job(p));assert r['completion']=='anomalous' and r['issues']


def test_valid_json_without_terminal_history_is_anomalous(tmp_path):
    p=tmp_path/'truncated.gz'
    with gzip.open(p,'wt') as f:
        json.dump(dict(summary=dict(success=False,ticks=1,decisions=1,stop='terminal'),
                       public_rows=[],events=[],decisions=[]),f)
    r=audit(job(p));assert r['completion']=='anomalous' and r['issues']


def test_content_hash_ignores_scene_id_only():
    from tools.cap_trace_value_transfer import content_digest
    assert content_digest({'scene_id':'a','goal':1}) == content_digest({'scene_id':'b','goal':1})
    assert content_digest({'scene_id':'a','goal':1}) != content_digest({'scene_id':'a','goal':2})
