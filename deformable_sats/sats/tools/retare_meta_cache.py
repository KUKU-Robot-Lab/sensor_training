#!/usr/bin/env python3
"""meta cache trial 의 로드셀 영점(tare) 오류를 재보정한다.

force = (loadcell_kg − baseline) × g 구조에서 baseline(무부하 영점)이 잘못 잡히면
fz_seq 전체가 상수만큼 shift 된다(무접촉 프레임의 force 가 0 이 아니게 됨).
이 도구는 무접촉(센서<threshold) 프레임의 force 중앙값으로 offset 을 추정해
fz_seq 에 더해 교정한다(원본은 .bak 백업). dataset 이 fz≤0 을 무접촉 처리하므로,
영점이 음수로 어긋나면 실접촉이 무접촉으로 오판돼 GT 가 오염되는 것을 바로잡는다.

사용::

    python -m sats.tools.retare_meta_cache <cache.pt> [--sensor-thr 0.5] [--apply]

--apply 없으면 추정·검증만(dry-run). 있으면 백업 후 교정 저장 + corrections.json 기록.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def _flatten(seqs):
    fz = np.concatenate([np.asarray(s["fz_seq"]).astype(float) for s in seqs])
    sensor = np.concatenate(
        [np.abs(np.asarray(s["sensor_seq"])).max(axis=1).astype(float) for s in seqs])
    return fz, sensor


def estimate_offset(seqs, sensor_thr: float) -> tuple[float, int]:
    """무접촉(센서<thr) 프레임의 force 중앙값 = 잘못된 영점. offset = −그 값."""
    fz, sensor = _flatten(seqs)
    nc = sensor < sensor_thr
    if nc.sum() < 50:
        raise ValueError(f"무접촉 프레임 부족(n={nc.sum()}) — thr 을 높이세요")
    return -float(np.median(fz[nc])), int(nc.sum())


def stability(seqs, offset: float) -> dict:
    """여러 임계값에서 교정 후 무접촉 force 가 0 근처인지(=상수 오프셋 확인)."""
    fz, sensor = _flatten(seqs)
    out = {}
    for thr in (0.2, 0.3, 0.5, 0.8):
        nc = sensor < thr
        if nc.sum() > 20:
            out[f"nc<{thr}_after"] = round(float(np.median(fz[nc]) + offset), 4)
    return out


def apply_offset(cache: dict, offset: float) -> None:
    for s in cache["sequences"]:
        s["fz_seq"] = (np.asarray(s["fz_seq"]).astype(np.float32) + np.float32(offset))


def main() -> None:
    p = argparse.ArgumentParser(description="meta cache 로드셀 영점 재보정")
    p.add_argument("cache", type=Path)
    p.add_argument("--sensor-thr", type=float, default=0.5)
    p.add_argument("--apply", action="store_true", help="실제 교정 저장(기본은 dry-run)")
    args = p.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    seqs = cache["sequences"]
    offset, n = estimate_offset(seqs, args.sensor_thr)
    print(f"trial: {cache.get('trial_id', args.cache.name)}")
    print(f"추정 tare offset = {offset:+.4f} N  (무접촉 n={n}, thr={args.sensor_thr})")
    print(f"교정 후 무접촉 force(≈0 이어야): {stability(seqs, offset)}")

    if not args.apply:
        print("\n[dry-run] --apply 를 붙이면 백업 후 교정 저장합니다.")
        return

    bak = args.cache.with_suffix(args.cache.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(args.cache, bak)
        print("백업:", bak)
    apply_offset(cache, offset)
    torch.save(cache, args.cache)
    print("교정 저장:", args.cache)

    # 교정 이력 기록(추적성)
    log = args.cache.parent / "corrections.json"
    entries = json.loads(log.read_text()) if log.exists() else []
    entries.append({
        "trial_id": cache.get("trial_id", args.cache.name),
        "cache": args.cache.name, "tare_offset_n": round(offset, 4),
        "sensor_thr": args.sensor_thr, "n_nocontact": n,
        "applied_at": datetime.now(timezone.utc).isoformat(),
    })
    log.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print("이력:", log)


if __name__ == "__main__":
    main()
