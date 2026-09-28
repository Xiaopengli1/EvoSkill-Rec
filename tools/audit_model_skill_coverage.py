from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recskill.evolution import audit_model_skill_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit torch_rechub model-to-skill coverage.")
    parser.add_argument("--repo-root", default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    issues = audit_model_skill_coverage(Path(args.repo_root))
    payload = [issue.to_dict() for issue in issues]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if not issues:
        print("Model skill coverage audit passed.")
        return
    for issue in issues:
        print(f"{issue.severity}\t{issue.model_name}\t{issue.issue}\t{issue.detail}")
    raise SystemExit(1 if any(issue.severity == "error" for issue in issues) else 0)


if __name__ == "__main__":
    main()
