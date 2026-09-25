from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.utils.reconstruction_baseline import (
    collect_baseline_rows,
    render_markdown_table,
    render_reclassification_notes,
)


def build_report(repo_root: Path) -> str:
    rows = collect_baseline_rows(repo_root)
    sections = [
        "# Reconstruction Baseline Report",
        "",
        "## Official Baseline Table",
        "",
        render_markdown_table(rows).rstrip(),
        "",
        "## Reclassification Notes",
        "",
        render_reclassification_notes(),
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the baseline report for reconstruction checklist 7.1.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(args.repo_root.resolve())
    if args.output is None:
        print(report)
        return
    args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
