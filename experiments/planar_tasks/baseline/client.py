"""Inspectable standard-library client for the development environment RPC."""
import json
import sys

_PREFIX = "__COIMAGINE_RPC__"


class DevelopmentEnv:
    def _call(self, method: str, **params):
        sys.__stdout__.write(_PREFIX + json.dumps({"method": method, **params}, allow_nan=False) + "\n")
        sys.__stdout__.flush()
        line = sys.__stdin__.readline(1048577)
        if not line:
            raise RuntimeError("Development environment disconnected")
        if len(line.encode()) > 1048576 or not line.endswith("\n"):
            raise RuntimeError("Invalid or oversized environment response")
        reply = json.loads(line)
        if not isinstance(reply, dict) or "ok" not in reply:
            raise RuntimeError("Invalid environment response envelope")
        if not reply["ok"]:
            error = reply.get("error", {})
            raise RuntimeError(f"Environment request failed: {error}")
        return reply["result"]

    def reset(self, task: str) -> dict:
        return self._call("reset", task=task)

    def step(self, action: dict) -> dict:
        return self._call("step", action=action)

    def observe(self) -> dict:
        return self._call("observe")

    def render(self) -> str:
        return self._call("render")
