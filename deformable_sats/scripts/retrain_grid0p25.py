#!/usr/bin/env python3
"""0.5mm(41²) 학습 모델들을 finer grid로 재학습 — 전처리→인덱스→학습 일괄.

--grid-step 으로 출력 격자 선택: 0.25mm(81²) / 0.1mm(201²). local_map 은 물리
범위(±3.5mm)를 유지하도록 격자에 맞춰 자동 환산, warm-start 는 해상도 이식(자동).

사용:
  로컬(5090):  .venv/bin/python scripts/retrain_grid0p25.py v5 v6 v7 --grid-step 0.1
  arm4090   :  .venv/bin/python scripts/retrain_grid0p25.py v8 v9 --grid-step 0.1

버전별 처리(순차):
  1) sensor_raw_bin/ecomesh_vN_xy1 없으면 prepare_learning_data 로 전처리
     (raw: skin_ws/raw_data/sats/ecomesh/vN/xy_1mm, depth d5=2.5·d10=3.5)
  2) trial_indices/ecomesh_vN_warm/dataset_index.json 없으면 생성(d5·d10 test 전부)
  3) train_e2e.train() 직접 호출 — v6 deploy config 템플릿 + grid 0.25/81,
     local_map 29(±3.5mm 물리 범위 유지), warm-start=자기 0.5 배포(없으면 v6 배포).
     0.5→0.25 는 local_map 최종 레이어 공간 업샘플 이식(train_e2e 자동).

산출: sats/training/runs/ecomesh_vN_deploy_g025/ (0.25mm) · _g01/ (0.1mm)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEPTH = {"d5": "2.5", "d10": "3.5"}


def _find_run_file(run_tail: str, filename: str) -> Path | None:
    """runs/ 에서 ecomesh_<tail> 과 <tail> 두 이름 모두 탐색.

    ★사용자가 UI 콤보 폭 때문에 run 폴더의 ecomesh_ 접두사를 제거(rename)한 PC 가
    있다 — 고정 경로는 그런 PC 에서 FileNotFoundError 로 죽는다.
    """
    for name in (f"ecomesh_{run_tail}", run_tail):
        f = _ROOT / "sats/training/runs" / name / filename
        if f.exists():
            return f
    return None


def template_cfg() -> Path:
    for tail in ("v6_deploy_all4", "v6_deploy_g025", "v5_deploy_all4"):
        f = _find_run_file(tail, "config.json")
        if f:
            return f
    raise FileNotFoundError("템플릿 config 없음 — runs/ 에 v6_deploy_all4(또는 g025) 필요")


def preprocess_if_needed(ver: str) -> Path:
    """sensor_raw_bin/ecomesh_vN_xy1 보장(없으면 merge 실행). 반환=raw_bin 디렉토리."""
    out = _ROOT / f"learning_data/sensor_raw_bin/ecomesh_{ver}_xy1"
    if out.exists() and any(out.rglob("*_merged.bin")):
        print(f"[{ver}] 전처리 스킵(존재): {out}")
        return out
    src = _ROOT / f"skin_ws/raw_data/sats/ecomesh/{ver}/xy_1mm"
    if not src.exists():
        raise FileNotFoundError(f"[{ver}] raw 없음: {src}")
    cmd = [sys.executable, str(_ROOT / "sats/preprocessing/prepare_learning_data.py"),
           "--source-root", str(_ROOT / "skin_ws/raw_data"),
           "--source-material", f"ecomesh/{ver}", "--material", f"ecomesh_{ver}",
           "--learning-root", str(_ROOT / "learning_data"), "--stage", "merge",
           "--depth-map", f"d5={DEPTH['d5']}", "--depth-map", f"d10={DEPTH['d10']}"]
    print(f"[{ver}] 전처리 실행: {' '.join(cmd[-8:])}")
    subprocess.run(cmd, check=True)
    if not any(out.rglob("*_merged.bin")):
        raise RuntimeError(f"[{ver}] 전처리 후에도 merged.bin 없음: {out}")
    return out


def build_index_if_needed(ver: str, raw_bin: Path) -> Path:
    """trial_indices/ecomesh_vN_warm/dataset_index.json 보장(전 trial 포함)."""
    idx_dir = _ROOT / f"learning_data/trial_indices/ecomesh_{ver}_warm"
    idx = idx_dir / "dataset_index.json"
    if idx.exists():
        print(f"[{ver}] 인덱스 스킵(존재): {idx}")
        return idx_dir
    trials = sorted(p.name.replace("_merged.bin", "")
                    for p in raw_bin.rglob("*_merged.bin"))
    if not trials:
        raise RuntimeError(f"[{ver}] merged.bin 없음: {raw_bin}")
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx.write_text(json.dumps({"trials": [{"trial_id": t} for t in trials]},
                              indent=2), encoding="utf-8")
    print(f"[{ver}] 인덱스 생성: {idx} ({len(trials)} trials)")
    return idx_dir


def warm_ckpt(ver: str) -> Path:
    own = _find_run_file(f"{ver}_deploy_all4", "best_model.pt")
    if own:
        return own
    fb = _find_run_file("v6_deploy_all4", "best_model.pt")
    if fb is None:
        raise FileNotFoundError(f"warm ckpt 없음: {ver}_deploy_all4 / v6_deploy_all4")
    return fb


def _grid_params(step: float) -> tuple[int, int, str]:
    """격자 간격 → (grid_size, local_map_size, run 접미사). local_map 은 ±3.5mm 유지."""
    grid_size = int(round(20.0 / step)) + 1
    local_map = 2 * int(round(3.5 / step)) + 1
    tag = {0.25: "g025", 0.1: "g01", 0.5: "g05"}.get(step, f"g{str(step).replace('.', 'p')}")
    return grid_size, local_map, tag


def train_version(ver: str, idx_dir: Path, epochs: int, step: float,
                  batch_size: int | None = None, size_input: bool = True) -> None:
    from sats.training.config import SATSConfig
    from sats.training.train_e2e import train

    raw = json.loads(template_cfg().read_text())
    valid = {f.name for f in fields(SATSConfig)}
    cfg_d = {k: v for k, v in raw.items() if k in valid}
    grid_size, local_map, tag = _grid_params(step)
    if not size_input:
        # ★size 조건 제거 실험: 원 논문에는 없는 d5/d10 명시 조건이 zero-shot 형상
        #   일반화(임의 인덴터)를 막는다는 가설 검증용. run 이름으로 구분(…ns).
        tag += "ns"
    run_name = f"ecomesh_{ver}_deploy_{tag}"
    cfg_d.update(
        grid_step_mm=step, grid_size=grid_size, local_map_size=local_map,  # ±3.5mm 유지
        gt_dir=str(idx_dir),
        dataset_index_path=str(idx_dir / "dataset_index.json"),
        run_name=run_name, out_dir=str(_ROOT / "sats/training/runs"),
        epochs=epochs,
        use_indenter_size_input=size_input,
    )
    if batch_size:                       # 0.1mm(201²)는 VRAM 24GB서 2048 OOM → 축소 필요
        cfg_d["batch_size"] = batch_size
    ckpt = warm_ckpt(ver)
    print(f"\n===== [{ver}] {step}mm 재학습 시작 → runs/{run_name} =====")
    print(f"  warm-start: {ckpt.parent.name} (해상도 이식 자동)")
    print(f"  epochs={epochs}  grid={grid_size}²@{step}mm  local_map={local_map}"
          f"  batch={cfg_d.get('batch_size')}")
    train(SATSConfig(**cfg_d), init_ckpt=str(ckpt))
    print(f"===== [{ver}] 완료 =====\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="finer grid 일괄 재학습")
    ap.add_argument("versions", nargs="+", help="예: v5 v6 v7 (로컬) / v8 v9 (arm4090)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--grid-step", type=float, default=0.25,
                    help="출력 격자 간격 mm (0.25=81² / 0.1=201²)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="미지정=템플릿(2048). 0.1mm는 VRAM 24GB서 OOM → 1024 권장")
    ap.add_argument("--skip-train", action="store_true", help="전처리·인덱스만")
    ap.add_argument("--no-size-input", action="store_true",
                    help="indenter 크기 조건(FiLM) 제거 — 형상 일반화 실험(run 접미사 ns)")
    args = ap.parse_args()
    for ver in args.versions:
        raw_bin = preprocess_if_needed(ver)
        idx_dir = build_index_if_needed(ver, raw_bin)
        if not args.skip_train:
            train_version(ver, idx_dir, args.epochs, args.grid_step, args.batch_size,
                          size_input=not args.no_size_input)
    print("전체 완료:", ", ".join(args.versions))


if __name__ == "__main__":
    main()
