"""Explicit opt-in DeepSeek transport; credentials never enter the agent workspace.

Reuses the existing bounded, no-redirect/no-retry official HTTPS transport, not
its legacy fixed-feedback synthesis loop. No network operation occurs on import.
"""
from .. import deepseek_agent


def deepseek_complete(payload,timeout):
    response=deepseek_agent.request(payload,timeout)
    if not response.get('ok'):
        raise RuntimeError('DeepSeek transport failed; no retry; billing may be unknown')
    return response['response']
