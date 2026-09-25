"""배포 모델 레지스트리 — 대시보드 가중치 탐색을 **폴더 규약**으로 통일.

규약(리포 루트 `runs/`, git clone 후 가중치만 넣으면 동작):
  runs/sats/<v*>/<run>/*.pt      S1·S3 SATS run — 같은 폴더에 config.json 필수
  runs/deform/<v*>/*.pt          S2·S3 변형 복원기(자가수록 체크포인트)
  runs/theta/<v*>/*.pt           S2 밴딩각 estimator (+ <name>.pt_ref_baseline.npy)

v* 폴더 안에 있는 .pt/.pth 는 이름 패턴과 무관하게 **전부** 후보로 올린다.
구 위치(sats/training/runs/*_deploy_*, sats/bending/runs/{deform_restorer*,estimator_*})도
폴백으로 계속 인식 — 두 머신의 기존 폴더를 옮기지 않아도 동작한다.
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_ROOT = _ROOT / "runs"
_PT_PATTERNS = ("*.pt", "*.pth")
_VER_RE = re.compile(r"^v\d+$")
# 클린 계보(g05nsc)를 기본 1순위로 — 이후 해상도/구버전 순
_RUN_ORDER = {"g05nsc": 0, "g05ns": 1, "g025": 2, "all4": 3, "g01": 4}


def _scan_pt(root: Path) -> list[Path]:
    """폴더 아래 모든 체크포인트 파일(재귀)."""
    if not root.is_dir():
        return []
    return sorted(p for pat in _PT_PATTERNS for p in root.rglob(pat) if p.is_file())


def _label(ver_dir: Path, path: Path) -> str:
    """콤보 표시용 축약 라벨 — v* 폴더 기준 상대경로에서 관용 파일명은 생략."""
    rel = path.relative_to(ver_dir).with_suffix("")
    parts = [p for p in rel.parts if p not in ("best_model", "best")]
    return "/".join(parts) or rel.parts[-1]


def _sig(path: Path) -> str:
    """중복 판정 시그니처 = 내용 md5 — 같은 가중치의 구 위치 사본을 콤보에 두 번
    띄우지 않는다. (mtime 기반은 git checkout 이 파일을 새로 쓰면 깨진다 — 실측.
    대상은 복원기·estimator 수백 KB 수준이라 해시 비용은 무시 가능.)"""
    import hashlib
    return hashlib.md5(path.read_bytes()).hexdigest()


def _version_dirs(category: str) -> list[Path]:
    base = DEPLOY_ROOT / category
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and _VER_RE.match(p.name))


def list_sensors() -> list[str]:
    """v* 드롭다운 후보 — 신규 폴더 규약 ∪ 구 배포 run 이름."""
    found = {d.name for cat in ("sats", "deform", "theta") for d in _version_dirs(cat)}
    for r in (_ROOT / "sats/training/runs").glob("*_deploy_*"):
        m = re.match(r"(?:ecomesh_)?(v\d+)_deploy_", r.name)
        if m and (r / "best_model.pt").exists():
            found.add(m.group(1))
    return sorted(found, key=lambda v: int(v[1:]))


def sats_runs(sensor: str) -> dict[str, Path]:
    """S1·S3: 그 센서의 SATS 후보 {라벨: 체크포인트 파일}. 클린 계보 우선 정렬.

    ★config.json 없는 폴더의 .pt 는 제외 — SATS 엔진이 로드할 수 없어 후보에
    올려봐야 선택 시 실패만 낳는다(콘솔 안내로 대신함).
    """
    out: dict[str, Path] = {}
    ver_dir = DEPLOY_ROOT / "sats" / sensor
    for p in _scan_pt(ver_dir):
        if (p.parent / "config.json").exists():
            out.setdefault(_label(ver_dir, p), p)
        else:
            print(f"[registry] config.json 없음 → 제외: {p.relative_to(_ROOT)}")
    base = _ROOT / "sats/training/runs"
    for pat in (f"ecomesh_{sensor}_deploy_*", f"{sensor}_deploy_*"):
        for r in base.glob(pat):
            if (r / "best_model.pt").exists() and (r / "config.json").exists():
                out.setdefault(r.name.rsplit("_", 1)[-1], r / "best_model.pt")
    return dict(sorted(out.items(),
                       key=lambda kv: (_RUN_ORDER.get(kv[0].rsplit("/", 1)[-1], 9), kv[0])))


def deform_restorers(sensor: str | None = None) -> dict[str, Path]:
    """S2·S3: 변형 복원기 후보 {라벨: 체크포인트 파일}.

    sensor 지정 시 그 v* 폴더만(+같은 센서의 구 폴더). 미지정이면 전부(v*/라벨).
    """
    out: dict[str, Path] = {}
    ver_dirs = ([DEPLOY_ROOT / "deform" / sensor] if sensor
                else _version_dirs("deform"))
    for ver_dir in ver_dirs:
        for p in _scan_pt(ver_dir):
            lab = _label(ver_dir, p) if sensor else f"{ver_dir.name}/{_label(ver_dir, p)}"
            out.setdefault(lab, p)
    seen = {_sig(p) for p in out.values()}
    for r in sorted((_ROOT / "sats/bending/runs").glob("deform_restorer*")):
        ck = r / "best.pt"
        if not ck.exists() or _sig(ck) in seen:
            continue
        tag = r.name.replace("deform_restorer_", "").replace("deform_restorer", "default")
        if sensor is None or tag in (sensor, "default"):
            out.setdefault(tag, ck)
    return out


def estimators() -> dict[str, Path]:
    """S2: theta estimator 후보 {라벨: 체크포인트}. 센서 구분 없이 전부 노출 —
    '다른 센서의 각도를 보여준다'가 S2 의 의도라 v* 로 거르지 않는다."""
    out: dict[str, Path] = {}
    troot = DEPLOY_ROOT / "theta"
    if troot.is_dir():
        for ver_dir in sorted(p for p in troot.iterdir() if p.is_dir()):
            for p in _scan_pt(ver_dir):
                out.setdefault(f"{ver_dir.name}/{_label(ver_dir, p)}", p)
    seen = {_sig(p) for p in out.values()}
    for r in sorted((_ROOT / "sats/bending/runs").glob("estimator_*")):
        ck = r / "best.pt" if r.is_dir() else r
        if r.is_dir() and not ck.exists():
            continue
        if r.is_file() and r.suffix != "":            # .npy·.json 등 부속 파일 제외
            continue
        if _sig(ck) not in seen:
            out.setdefault(r.name.replace("estimator_", ""), ck)
    return out


def default_estimator() -> Path | None:
    """기동 시 기본 estimator — v6_2(클린) 우선, 없으면 아무거나, 그것도 없으면 None.

    ★None 이어도 대시보드는 켜져야 한다(S2 에서 나중에 선택) — 여기서 예외를 내면
    가중치 없는 fresh clone 에서 기동 자체가 막힌다.
    """
    ests = estimators()
    for key in ests:
        if "v6_2" in key:
            return ests[key]
    return next(iter(ests.values()), None)


def resolve_sats_run(sensor: str | None, explicit: str | None) -> Path:
    """센서 버전 → SATS 체크포인트 경로(파일 또는 run 폴더). --run-dir 우선."""
    if explicit:
        return Path(explicit)
    target = sensor or "v6"
    runs = sats_runs(target)
    if not runs:
        raise SystemExit(
            f"[{target}] SATS run 을 찾지 못했습니다. 확인한 위치:\n"
            f"  {DEPLOY_ROOT / 'sats' / target}\n"
            f"  {_ROOT / 'sats/training/runs'}/(ecomesh_)?{target}_deploy_*\n"
            f"  → --run-dir 로 직접 지정하세요.")
    return next(iter(runs.values()))
