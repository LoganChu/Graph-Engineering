"""Importing this package registers every orchestrator under its CLI name.

`single_shot` has no third-party dependency and always loads. `loop` and `graph`
need LangChain and LangGraph, which live behind an optional extra -- a missing
extra should leave the historical single-shot matrix working, so the import
failure is swallowed here and only surfaces if you ask for those by name.
"""

from . import single_shot  # noqa: F401

try:  # pragma: no cover - depends on optional extra
    from . import graph, loop  # noqa: F401
except ImportError:
    pass
