from __future__ import annotations

import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from recskill.evolution.ctr_workflow import main


if __name__ == "__main__":
    raise SystemExit(main())
