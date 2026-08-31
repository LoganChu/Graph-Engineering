"""Importing this package registers every orchestrator under its CLI name.

`single_shot` and `fanout` have no third-party dependency and always load --
which matters for `fanout` specifically, since it is the ablation the iterative
orchestrators are read against and a control that needs an optional extra is a
control you cannot always run.

`loop`, `graph` and `plan_execute` need LangChain and LangGraph, which live
behind an optional extra. A missing extra should leave the historical
single-shot matrix working, so the import failure is swallowed here and only
surfaces if you ask for those by name.
"""

from . import fanout, single_shot  # noqa: F401

try:  # pragma: no cover - depends on optional extra
    from . import graph, loop, plan_execute  # noqa: F401
except ImportError:
    pass
