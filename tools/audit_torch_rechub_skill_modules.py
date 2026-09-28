from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recskill.evolution.skill_implementation_audit import audit_torch_rechub_skill_implementations


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Torch-RecHub model abilities against recskill/skills implementations only.")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="Print machine-readable issues.")
    args = parser.parse_args()

    issues = audit_torch_rechub_skill_implementations(args.repo_root)
    if args.json:
        print(json.dumps([issue.to_dict() for issue in issues], indent=2, sort_keys=True))
    else:
        if not issues:
            print("Torch-RecHub model ability skill audit passed.")
            return
        for issue in issues:
            print(f"{issue.severity}\t{issue.model_name}\t{issue.issue}\t{issue.detail}")
    if any(issue.severity == "error" for issue in issues):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
