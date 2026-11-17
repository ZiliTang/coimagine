import io
import tarfile

import pytest

from tools import program_trace_transfer as T


def tiny_bundle(tmp_path):
    bundle = tmp_path / 'new_bundle'
    bundle.mkdir()
    (bundle / 'data.txt').write_text('evidence')
    T.P.save(bundle / 'manifest.json', dict(version=T.VERSION, files={'data.txt': T.P.sha(bundle / 'data.txt')}))
    return bundle


def test_relative_reference_preserves_dependency_and_rejects_escape(tmp_path):
    bundle = tmp_path / 'new_repo' / 'bundle'
    dep = tmp_path / 'different_root' / 'priority'
    assert T.resolve(dict(root='bundle', path='a/file'), bundle, dep) == bundle / 'a/file'
    assert T.resolve(dict(root='priority', path='a/file'), bundle, dep) == dep / 'a/file'
    for ref in [dict(root='old_machine', path='a'), dict(root='bundle', path='../escape'),
                dict(root='priority', path='/home/zilitang/original')]:
        with pytest.raises(ValueError):
            T.resolve(ref, bundle, dep)


def test_inventory_rejects_extra_and_changed_payload(tmp_path):
    bundle = tiny_bundle(tmp_path)
    T.check_inventory(bundle)
    (bundle / 'extra').write_text('unexpected')
    with pytest.raises(ValueError, match='inventory'):
        T.check_inventory(bundle)
    (bundle / 'extra').unlink()
    (bundle / 'data.txt').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        T.check_inventory(bundle)


def test_export_cannot_write_into_existing_evidence(tmp_path):
    dep = tmp_path / 'priority'
    dep.mkdir()
    with pytest.raises(ValueError, match='frozen inputs'):
        T.check_export_destination(dep / 'new_export', dep)
    with pytest.raises(ValueError, match='frozen inputs'):
        T.check_export_destination(T.STUDY / 'new_export', dep)
    existing = tmp_path / 'exists'
    existing.mkdir()
    with pytest.raises(FileExistsError):
        T.check_export_destination(existing, dep)
    T.check_export_destination(tmp_path / 'new_export', dep)


def test_archive_roundtrip_and_refusal_to_overwrite(tmp_path):
    bundle = tiny_bundle(tmp_path)
    archive = tmp_path / 'handoff.tar.gz'
    result = T.pack(bundle, archive)
    assert result['files'] == 2
    assert T.audit_archive(archive, bundle) == 2
    assert result['sha256'] in archive.with_name(archive.name + '.sha256').read_text()
    with pytest.raises(ValueError):
        T.pack(bundle, archive)


def test_archive_rejects_unlisted_and_missing_members(tmp_path):
    bundle = tiny_bundle(tmp_path)
    archive = tmp_path / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as f:
        member = tarfile.TarInfo(bundle.name + '/../outside')
        member.size = 1
        f.addfile(member, io.BytesIO(b'x'))
    with pytest.raises(ValueError):
        T.audit_archive(archive, bundle)
    with tarfile.open(archive, 'w:gz'):
        pass
    with pytest.raises(ValueError, match='missing'):
        T.audit_archive(archive, bundle)


def test_duplicate_reference_cannot_hide_inconsistent_hash(tmp_path):
    bundle = tiny_bundle(tmp_path)
    dep = tmp_path / 'priority'
    dep.mkdir()
    T.P.save(dep / 'manifest.json', dict(version=T.P.VERSION, files={}))
    T.P.save(bundle / 'dependency.json', dict(manifest_sha256=T.P.sha(dep / 'manifest.json')))
    good = dict(root='bundle', path='data.txt', sha256=T.P.sha(bundle / 'data.txt'))
    bad = dict(good, sha256='0' * 64)
    T.P.save(bundle / 'index.json', dict(original270=[dict(trajectory=good), dict(trajectory=bad)],
             repair_suffixes=[], replays=[], handoff_states=[], models=[]))
    files = {p.name: T.P.sha(p) for p in bundle.iterdir() if p.name != 'manifest.json'}
    T.P.save(bundle / 'manifest.json', dict(version=T.VERSION, files=files))
    with pytest.raises(ValueError, match='index reference checksum'):
        T.verify(bundle, dep, data_only=True)
