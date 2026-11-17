"""Use frozen executors with a separately versioned, backend-independent knowledge pack."""
from pathlib import Path
import json
from experiments.planar_tasks.baseline import session as free
from experiments.planar_tasks.option_search_v3 import session as search
from . import knowledge


class KnowledgeMixin:
    def _active(self):
        super()._active()
        knowledge.verify_session(self.path)

    def write_file(self, path, text):
        if path == 'option_guide.md':
            raise ValueError('public skill knowledge is host-owned')
        return super().write_file(path, text)

    def submit(self, files, contents=None):
        profile = knowledge.verify_session(self.path)
        result = super().submit(files, contents)
        result['public_skill_knowledge'] = profile
        result['public_knowledge_sha256'] = self.state['public_knowledge_sha256']
        free.write_json(self.path / 'frozen/manifest.json', result)
        return result


class FreeSession(KnowledgeMixin, free.Session):
    @classmethod
    def create(cls, path, config, **kwargs):
        return create_session(path, config, method='free', **kwargs)


class SearchSession(KnowledgeMixin, search.Session):
    @classmethod
    def create(cls, path, config, **kwargs):
        return create_session(path, config, method='option_search_v3', **kwargs)


def create_session(path, config, *, method='option_search_v3', **kwargs):
    knowledge.verify()
    if method not in ('free', 'option_search_v3'):
        raise ValueError('method must be free or option_search_v3')
    parent = free.Session if method == 'free' else search.Session
    session = parent.create(path, config, **kwargs)
    try:
        knowledge.install(session, method)
    finally:
        session.close()
    return open_session(path, **({'world_factory': kwargs['world_factory']} if 'world_factory' in kwargs else {}))


def open_session(path, **kwargs):
    profile = knowledge.verify_session(path)
    cls = {'free': FreeSession, 'option_search_v3': SearchSession}[profile['method']]
    return cls(path, **kwargs)


def verify_frozen(path):
    path = Path(path)
    profile = knowledge.verify_session(path)
    module = free if profile['method'] == 'free' else search
    manifest = module.verify_frozen(path)
    if (manifest.get('public_skill_knowledge') != profile
            or manifest.get('public_knowledge_sha256') != free.seal.digest(path / 'public_knowledge.json')):
        raise ValueError('frozen knowledge provenance mismatch')
    return manifest
