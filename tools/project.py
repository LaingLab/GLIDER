"""Inspect, check and adopt a GLIDER project folder.

    python tools/project.py doctor "Z:/.../videos"
    python tools/project.py plan   "Z:/.../videos"
    python tools/project.py adopt  "Z:/.../videos"      # asks first
    python tools/project.py revert "Z:/.../videos"

``doctor`` reports what is detectably wrong and changes nothing. ``plan`` shows
the complete source-to-destination mapping for adoption, also changing nothing.
``adopt`` carries a plan out, and refuses outright if the plan has a single
collision - the whole point is that a folder never ends up half-moved.

Adoption is reversible: it writes ``adopt_reversal.json`` before moving
anything, and ``revert`` replays it backwards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glider.core.adopt import (  # noqa: E402
    REVERSAL_NAME,
    apply_plan,
    plan_adopt,
    revert,
)
from glider.core.doctor import doctor, format_report  # noqa: E402
from glider.core.project import Project  # noqa: E402


def cmd_doctor(args: argparse.Namespace) -> int:
    project = Project.load(args.root)
    findings = doctor(project)
    print(format_report(findings, project=project))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    plan = plan_adopt(args.root)
    print(plan.describe())
    return 0 if plan.is_safe else 1


def cmd_adopt(args: argparse.Namespace) -> int:
    plan = plan_adopt(args.root)
    print(plan.describe())
    if not plan.is_safe:
        print("\nNothing moved. Resolve the conflicts above and run again.")
        return 1
    if not plan.moves:
        print("\nAlready in the canonical layout.")
        return 0
    if not args.yes:
        # Adoption moves irreplaceable data on a network share. The default is
        # to ask, every time.
        answer = input(f"\nMove {len(plan.moves)} files? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Nothing moved.")
            return 1

    result = apply_plan(plan)
    print(f"\nmoved {len(result.moved)}, already done {len(result.already_done)}")
    if result.reversal_path is not None:
        print(f"reversal manifest: {result.reversal_path}")
    if not result.ok:
        print(f"STOPPED: {result.error}")
        print("Run 'revert' to put back what moved.")
        return 1
    return 0


def cmd_revert(args: argparse.Namespace) -> int:
    path = Path(args.root)
    if path.is_dir():
        path = path / REVERSAL_NAME
    result = revert(path)
    print(f"put back {len(result.moved)}, already back {len(result.already_done)}")
    if not result.ok:
        print(f"STOPPED: {result.error}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler, help_text in (
        ("doctor", cmd_doctor, "report what is detectably wrong; changes nothing"),
        ("plan", cmd_plan, "show how adoption would move files; changes nothing"),
        ("adopt", cmd_adopt, "move the folder into the canonical layout"),
        ("revert", cmd_revert, "undo an adoption from its reversal manifest"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("root", help="the project folder")
        if name == "adopt":
            p.add_argument("--yes", action="store_true", help="skip the confirmation")
        p.set_defaults(handler=handler)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
