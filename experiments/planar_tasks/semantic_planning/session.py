"""Thin authoring_v2 adapter; preserve historical seals and fix mixed log reading."""
import json
from experiments.planar_tasks.authoring_v2 import session as authoring


class Session(authoring.SearchSession):
    def _observations(self, record_id, branch=None):
        if record_id in self.state.get('batches', []):
            return super()._observations(record_id, branch)
        if branch is not None:
            raise ValueError('branch only applies to batches')
        path = self._test_path(record_id) / 'sensors.jsonl'
        if not path.exists():
            return []
        observations = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            # The frozen runner alternates request and response rows. Responses
            # contain actions, not another observation. Never double-count them.
            if set(row) == {'response'}:
                continue
            observations.append(row['request']['observation'])
        return observations


def create_session(path, config, **kwargs):
    """Use the default v2 creator and every original budget/knowledge check."""
    original = authoring.create_session(path, config, method='option_search_v3', **kwargs)
    original.close()
    return Session(path, **({'world_factory': kwargs['world_factory']} if 'world_factory' in kwargs else {}))


def open_session(path, **kwargs):
    return Session(path, **kwargs)


verify_frozen = authoring.verify_frozen
