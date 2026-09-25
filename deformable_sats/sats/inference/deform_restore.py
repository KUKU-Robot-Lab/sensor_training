"""각도-프리 변형 복원기 — 데모 추론 래퍼.

학습(`sats.bending.train_deform_restorer`)이 만든 latent restorer 를 데모에서 쓰기 위한
얇은 껍데기다. 하는 일은 하나: **16채널 잔차(pct)에서 변형분만큼 뺀다.**

    restored = pct - offset(z(pct))

SATS 입력은 원래 flat 무접촉 baseline 대비 잔차로 학습됐다. 변형이 생기면 baseline 이
이동해 유령이 뜨는데, 그 이동량을 빼주면 SATS 는 학습 때 보던 분포를 다시 받는다.
즉 SATS 를 손대지 않고 **입력만 되돌리는** 구조다(동결 SATS).

★파손 taxel 마스킹은 체크포인트에 저장된 목록을 그대로 쓴다. 학습과 다른 채널 구성으로
추론하면 복원기가 존재하지 않는 taxel 을 근거로 오프셋을 추정한다.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

_DEFAULT_CKPT = Path(__file__).resolve().parents[2] / "sats/bending/runs/deform_restorer/best.pt"


class DeformRestorer:
    """변형된 상태의 pct 창을 flat 기준으로 되돌린다."""

    def __init__(self, ckpt_path: str | Path = _DEFAULT_CKPT, device: str = "cpu") -> None:
        from sats.bending.baseline_restorer import BaselineRestorer
        from sats.bending.config import BendingConfig

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"변형 복원기 체크포인트 없음: {ckpt_path}\n"
                f"  → python -m sats.bending.train_deform_restorer --deform-root ... 로 먼저 학습")
        ck = torch.load(ckpt_path, map_location=device)
        if ck.get("restorer_mode") != "latent":
            raise ValueError(f"latent 모드 체크포인트가 아님: {ck.get('restorer_mode')!r}")
        cfg = replace(BendingConfig(), restorer_mode="latent",
                      latent_dim=int(ck.get("latent_dim", 32)))
        self.model = BaselineRestorer(cfg).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.device = device
        self.latent_dim = cfg.latent_dim
        self.window_size = cfg.window_size
        # 0-based. 학습과 반드시 동일해야 하므로 체크포인트 값을 신뢰한다.
        self.dead = tuple(int(c) - 1 for c in ck.get("dead_channels_1based", []))
        self.sensor = ck.get("sensor")

    def describe(self) -> str:
        dead = ",".join(f"S{c + 1:02d}" for c in self.dead) or "없음"
        return (f"변형 복원기 latent_dim={self.latent_dim} "
                f"sensor={self.sensor or '?'} 마스킹={dead}")

    def mask(self, pct_window: np.ndarray) -> np.ndarray:
        """파손 taxel 을 0(= baseline 대비 변화 없음)으로 고정한 사본."""
        if not self.dead:
            return np.asarray(pct_window, np.float32)
        out = np.array(pct_window, np.float32, copy=True)
        out[..., list(self.dead)] = 0.0
        return out

    @torch.no_grad()
    def restore(self, pct_window: np.ndarray) -> np.ndarray:
        """pct[W,16] 또는 [B,W,16] → 변형분을 뺀 pct(같은 shape)."""
        x = self.mask(pct_window)
        single = x.ndim == 2
        t = torch.from_numpy(x[None] if single else x).to(self.device)
        y = self.model(t).cpu().numpy().astype(np.float32)
        return y[0] if single else y


def try_load(ckpt_path: str | Path = _DEFAULT_CKPT, device: str = "cpu"
             ) -> tuple[DeformRestorer | None, str]:
    """(복원기, 상태문구). 실패해도 데모 전체가 죽지 않도록 예외를 문구로 바꾼다."""
    try:
        r = DeformRestorer(ckpt_path, device)
    except Exception as e:
        return None, f"복원기 없음 — {e}"
    return r, r.describe()


def _patch_cli() -> int:
    """구 체크포인트에 마스킹 메타데이터 주입(재학습 불필요).

    메타 저장 기능 이전에 학습된 best.pt 는 dead_channels/sensor 가 없어
    추론에서 마스킹이 빠진다(sensor=? 마스킹=없음). 모델 가중치는 마스킹된
    입력으로 학습되어 그대로 유효하므로 메타만 채우면 된다.

    사용: python -m sats.inference.deform_restore <ckpt> --dead-channels 7,11,15 --sensor v7
    """
    import argparse
    ap = argparse.ArgumentParser(description="deform 복원기 체크포인트 메타 보정")
    ap.add_argument("ckpt")
    ap.add_argument("--dead-channels", required=True, help="1-based, 예 '7,11,15'")
    ap.add_argument("--sensor", required=True, help="예 v7")
    a = ap.parse_args()
    path = Path(a.ckpt)
    ck = torch.load(path, map_location="cpu")
    ck["dead_channels_1based"] = [int(x) for x in a.dead_channels.split(",") if x]
    ck["sensor"] = a.sensor
    torch.save(ck, path)
    print(f"보정 완료: {path}")
    print(f"  sensor={ck['sensor']}  마스킹={ck['dead_channels_1based']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_patch_cli())
