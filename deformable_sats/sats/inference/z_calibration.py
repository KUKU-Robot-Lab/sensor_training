"""z(압입깊이) 보정 — SATS 맵 peak_val → z_depth(mm) 선형 회귀 LUT (A1 방식).

모델은 압력맵만 출력하므로 z를 직접 못 낸다. v6 학습데이터(merged bin + GT z_depth_mm)로
인덴터 지름별 z_depth ≈ a·peak_val + b 를 적합해 저장하고, 추론 때 맵 peak로 z를 근사한다.

성립성(v6): d5 R²≈0.87(양호), d10 R²≈0.59(근사). z는 근사값으로 표시할 것.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_JSON = Path(__file__).resolve().parent / "z_calibration_v6.json"


@dataclass(frozen=True)
class ZFit:
    """z_depth(mm) = slope·peak_val + intercept (지름별)."""
    slope: float
    intercept: float
    r2: float
    z_min: float
    z_max: float

    def z_from_peak(self, peak_val: float) -> float:
        z = self.slope * float(peak_val) + self.intercept
        return float(np.clip(z, 0.0, self.z_max * 1.2))


class ZCalibration:
    """지름별 ZFit 모음. 없는 지름은 최근접 지름으로 대체."""

    def __init__(self, fits: dict[float, ZFit]):
        if not fits:
            raise ValueError("z calibration is empty")
        self.fits = fits

    @classmethod
    def load(cls, path: str | Path = DEFAULT_JSON) -> "ZCalibration":
        data = json.loads(Path(path).read_text())
        fits = {float(k): ZFit(**v) for k, v in data.items()}
        return cls(fits)

    def _nearest(self, diameter_mm: float) -> ZFit:
        key = min(self.fits, key=lambda d: abs(d - float(diameter_mm)))
        return self.fits[key]

    def z_from_peak(self, peak_val: float, diameter_mm: float) -> float:
        return self._nearest(diameter_mm).z_from_peak(peak_val)


def build_z_calibration(sats_run: str | Path,
                        raw_root: str | Path = "learning_data/sensor_raw_bin/ecomesh_v6_xy1",
                        patterns: dict[float, str] | None = None,
                        out_json: str | Path = DEFAULT_JSON,
                        device: str = "cuda", fz_min: float = 0.2, stride: int = 15) -> ZCalibration:
    """v6 학습데이터로 지름별 z_depth≈a·peak_val+b 적합 후 저장. (오프라인 1회 실행)"""
    import torch
    from scipy import stats
    from sats.bending.pipeline import load_frozen_sats
    from sats.bending.eval_pipeline import _sats_map
    from sats.preprocessing.merged_bin import merged_bin_to_frame
    from sats.training.dataset import _load_baseline

    patterns = patterns or {5.0: "*d5*test1*", 10.0: "*d10*test1*"}
    skin = [f"s{i}" for i in range(1, 17)]
    W = 10
    sats = load_frozen_sats(sats_run, device)
    fits: dict[float, ZFit] = {}
    for diam, patt in patterns.items():
        matches = glob.glob(str(Path(raw_root) / f"**/{patt}_merged.bin"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"no merged bin for diameter {diam}: {patt}")
        b = Path(matches[0])
        df = merged_bin_to_frame(b)
        base = np.asarray(_load_baseline(Path(glob.glob(str(b.parent / "*_baseline.json"))[0]), merged_bin=b), float)
        pct = ((df[skin].to_numpy(float) - base) / np.where(np.abs(base) < 1e-9, 1e-9, base) * 100).astype(np.float32)
        fz = df["Fz"].to_numpy(); zdep = df["z_depth_mm"].to_numpy()
        e = np.where(fz > fz_min)[0]; e = e[e >= W - 1][::stride]
        win = np.stack([pct[i - W + 1:i + 1] for i in e])
        pm = np.concatenate([_sats_map(sats, torch.from_numpy(win[i:i + 2048]).to(device)).cpu().numpy()
                             for i in range(0, len(win), 2048)])
        peakv = pm.reshape(len(pm), -1).max(1); z = zdep[e]
        r = stats.linregress(peakv, z)
        fits[diam] = ZFit(slope=float(r.slope), intercept=float(r.intercept), r2=float(r.rvalue ** 2),
                          z_min=float(z.min()), z_max=float(z.max()))
        print(f"[z-cal d{diam:g}] z = {r.slope:.4g}·peak + {r.intercept:.4g}  R²={r.rvalue**2:.3f}  (n={len(e)})")
    Path(out_json).write_text(json.dumps({str(k): v.__dict__ for k, v in fits.items()}, indent=2))
    print(f"saved: {out_json}")
    return ZCalibration(fits)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="v6 z 보정 LUT 생성")
    p.add_argument("--sats-run", default="sats/training/runs/ecomesh_v6_deploy_all4")
    p.add_argument("--device", default="cuda")
    build_z_calibration(p.parse_args().sats_run, device=p.parse_args().device)
