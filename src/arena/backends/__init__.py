"""Importing this package registers every backend under its CLI name.

The vector backend needs an optional dependency; a missing extra should not stop
the other five from running, so its import failure is swallowed here and
surfaces only if you actually ask for it by name.
"""

from . import agent_notes, baselines, entity_graph, lexical, temporal_graph  # noqa: F401

try:  # pragma: no cover - depends on optional extra
    from . import vector  # noqa: F401
except Exception:
    pass
