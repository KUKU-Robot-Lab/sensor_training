"""실시간 밴딩 추론 — estimator(theta) + restorer(flat 복원). 버클 방식(D).

★입력 규약(중요): estimator 는 **원시 센서 윈도우를 표준화**해 받는다(pct 아님).
BMP384는 절대 기압을 재므로 raw 절대값이 날씨/온도/유닛에 따라 달라진다. estimator는
학습 stats(mean~6.7M, std~수만)로 절대 raw를 표준화하는데, std가 mean의 ~1%라 baseline이
0.5%만 어긋나도 ~1σ OOD → theta가 포화(예: flat인데 142°). 따라서 데모의 raw를 그대로
쓰면 안 되고, **학습 참조 baseline(ref)에 상대패턴(pct)을 얹어 재앵커**한다:
  raw_est = ref × (1 + pct/100).
이러면 데모 당일 기압과 무관하게 학습과 동일 분포가 된다(실측 검증: 재앵커=in-dist 일치).
ref 는 estimator_ckpt 옆 `<ckpt>_ref_baseline.npy`(v6 학습 trial baseline 채널평균).

restorer 는 pct(+theta) 를 받아 flat 등가 pct 를 돌려준다(동결 SATS 입력).
밴딩 방식은 v6 buckling(Y구동 δ=Y−18)로 학습됨. 순수 jig-bend 는 신호가 약해 포화 →
데모는 buckling 지그 전제([[v6-test-eval]]).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sats.bending.config import BendingConfig
from sats.bending.train_bending import load_estimator


def pct_to_raw(pct_window: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """pct 윈도우[W,16] + baseline[16] → 원시 센서 윈도우[W,16] 복원."""
    return (baseline[None, :] * (1.0 + pct_window / 100.0)).astype(np.float32)


class BendingInference:
    """estimator theta + (선택) restorer 복원. window 크기는 estimator cfg 기준.

    ref_baseline 있으면 pct→raw 재앵커에 사용(기압 무관). 없으면 데모 baseline 필요.
    """

    def __init__(self, estimator_ckpt: str | Path, device: str = "cpu",
                 restorer=None, cfg: BendingConfig | None = None) -> None:
        self.device = device
        self.est, self.stats = load_estimator(estimator_ckpt, device=device)
        self.cfg = cfg or BendingConfig()
        self.W = self.cfg.window_size
        self._mean = torch.tensor(np.asarray(self.stats.mean), device=device)
        self._std = torch.tensor(np.asarray(self.stats.std), device=device)
        self.restorer = restorer
        # 학습 참조 baseline(재앵커용) — <ckpt>_ref_baseline.npy
        ref_path = Path(str(estimator_ckpt) + "_ref_baseline.npy")
        self.ref_baseline = np.load(ref_path).astype(np.float32) if ref_path.exists() else None

    @torch.no_grad()
    def theta_from_raw(self, raw_window: np.ndarray) -> float:
        """원시 윈도우[W,16] → 밴딩각 theta(deg, signed)."""
        std = (torch.from_numpy(raw_window.astype(np.float32)).to(self.device) - self._mean) / self._std
        L = torch.full((1,), std.shape[0], dtype=torch.long, device=self.device)
        deg = self.est(std.unsqueeze(0), L)
        return float(deg.reshape(-1)[0].cpu())

    def theta_from_pct(self, pct_window: np.ndarray, demo_baseline: np.ndarray | None = None) -> float:
        """pct 윈도우[W,16] → theta. 학습 참조 baseline(ref)에 재앵커(기압 무관).

        ref_baseline 없으면 demo_baseline 로 폴백(구 동작, 기압 민감 — 권장 안 함).
        """
        anchor = self.ref_baseline if self.ref_baseline is not None else demo_baseline
        if anchor is None:
            raise ValueError("재앵커 baseline 없음: ref_baseline.npy 또는 demo_baseline 필요")
        return self.theta_from_raw(pct_to_raw(pct_window, np.asarray(anchor, np.float32)))

    @torch.no_grad()
    def restore(self, pct_window: np.ndarray, theta_deg: float) -> np.ndarray:
        """pct 윈도우[W,16] + theta → flat 등가 pct[W,16] (restorer 없으면 그대로)."""
        if self.restorer is None:
            return pct_window.astype(np.float32)
        x = torch.from_numpy(pct_window.astype(np.float32)).to(self.device).unsqueeze(0)
        deg = torch.full((1,), float(theta_deg), device=self.device)
        return self.restorer(x, deg)[0].cpu().numpy().astype(np.float32)


def load_restorer(sats_run: str | Path, bending_dir: str | Path,
                  device: str = "cpu", cfg: BendingConfig | None = None,
                  epochs: int = 120):
    """v6 buckling 밴딩으로 restorer 학습(데모 셋업 시 1회, ~1분)."""
    from sats.bending.eval_pipeline import _train_pct_restorer
    from sats.bending.pipeline import load_frozen_sats
    cfg = cfg or BendingConfig()
    sats = load_frozen_sats(sats_run, device)
    trials = sorted(p.stem for p in Path(bending_dir).glob("*.npz"))
    if not trials:
        raise FileNotFoundError(f"밴딩 npz 없음: {bending_dir}")
    return _train_pct_restorer(sats, Path(bending_dir), trials, cfg, device, epochs, 1e-3)
