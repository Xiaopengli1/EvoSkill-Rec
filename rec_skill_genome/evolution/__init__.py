from __future__ import annotations

import importlib
import sys

from recskill.evolution import *  # noqa: F401,F403

_SUBMODULES = [
    "compiler",
    "evolution_memory",
    "exceptions",
    "genome",
    "mutations",
    "open_ended",
    "planners",
    "skill_library",
    "verification",
]

for _name in _SUBMODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"recskill.evolution.{_name}")
