#!/usr/bin/env python3
"""구 위치의 배포 가중치를 대시보드 폴더 규약(runs/)으로 복사한다.

  sats/training/runs/(ecomesh_)?<v*>_deploy_<tag>/   → runs/sats/<v*>/<tag>/{best_model.pt, config.json}
  sats/bending/runs/deform_restorer_<v*>/best.pt     → runs/deform/<v*>/restorer.pt
  sats/bending/runs/estimator_<name>/best.pt(+ref)   → runs/theta/<v*|misc>/estimator_<name>.pt(+ref)

멱등: 대상이 이미 있고 크기가 같으면 건너뛴다. 원본은 옮기지 않는다(학습·sync
스크립트가 구 경로를 계속 쓰므로) — 대시보드 탐색만 새 규약을 1순위로 본다.

사용: .venv/bin/python scripts/stage_deploy_runs.py [--root <repo>]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_DEPLOY_RE = re.compile(r"^(?:ecomesh_)?(v\d+)_deploy_(.+)$")
_EST_VER_RE = re.compile(r"^estimator_(v\d+)")


def _copy(src: Path, dst: Path) -> bool:
    """내용이 같으면 스킵. 복사했으면 True.

    ★크기 비교만으로는 부족 — 재학습본은 크기가 같고 내용만 다른 경우가 실재
    (deform restorer 36053B 동일, md5 상이). filecmp 로 바이트 비교한다."""
    import filecmp
    if not src.exists():
        return False
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def stage_sats(root: Path) -> int:
    n = 0
    for run in sorted((root / "sats/training/runs").glob("*_deploy_*")):
        m = _DEPLOY_RE.match(run.name)
        if not m or not (run / "best_model.pt").exists():
            continue
        sensor, tag = m.groups()
        dst = root / "runs/sats" / sensor / tag
        copied = _copy(run / "best_model.pt", dst / "best_model.pt")
        copied |= _copy(run / "config.json", dst / "config.json")
        if copied:
            print(f"  sats  {run.name} → runs/sats/{sensor}/{tag}/")
            n += 1
    return n


def stage_deform(root: Path) -> int:
    n = 0
    for run in sorted((root / "sats/bending/runs").glob("deform_restorer_v*")):
        sensor = run.name.replace("deform_restorer_", "")
        if _copy(run / "best.pt", root / "runs/deform" / sensor / "restorer.pt"):
            print(f"  deform {run.name} → runs/deform/{sensor}/restorer.pt")
            n += 1
    return n


def stage_theta(root: Path) -> int:
    n = 0
    for run in sorted((root / "sats/bending/runs").glob("estimator_*")):
        if not run.is_dir() or not (run / "best.pt").exists():
            continue
        m = _EST_VER_RE.match(run.name)
        ver = m.group(1) if m else "misc"
        dst = root / "runs/theta" / ver / f"{run.name}.pt"
        copied = _copy(run / "best.pt", dst)
        # BendingInference 는 <ckpt 경로>+"_ref_baseline.npy" 를 찾는다 — 이름을 맞춰 복사.
        copied |= _copy(run / "best.pt_ref_baseline.npy",
                        dst.parent / f"{dst.name}_ref_baseline.npy")
        if copied:
            print(f"  theta {run.name} → runs/theta/{ver}/{dst.name}")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    a = ap.parse_args()
    root = a.root.resolve()
    if not (root / "sats").is_dir():
        print(f"리포 루트가 아님: {root}", file=sys.stderr)
        return 1
    total = stage_sats(root) + stage_deform(root) + stage_theta(root)
    print(f"완료 — 갱신 {total}건 (기존 동일 파일은 스킵). 루트: {root / 'runs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
