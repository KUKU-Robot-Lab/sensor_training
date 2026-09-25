"""
sats/inference/inference_engine.py

학습된 SATS 모델을 사용해 실시간 추론을 수행하는 엔진.

입력  : ndarray[window_size, 16]  (s_norm 슬라이딩 윈도우)
출력  : ndarray[41, 41]           (정제된 압력 맵, 학습 단위 N/mm²×100)
        + peak 위치 (x_mm, y_mm, peak_val)
        + Fz 추정값 (N)
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sats.training.config import SATSConfig


# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────

# 기본 grid 상수 (하위호환·기본값 표시용). 실제 좌표/면적은 로드한 config 기반
# 인스턴스 속성(self.grid_*)을 사용하므로, 0.1/0.25mm 등 임의 출력 해상도 모델도 지원.
GRID_SIZE    = 41
GRID_MIN_MM  = -10.0
GRID_MAX_MM  = 10.0
GRID_STEP_MM = 0.5
TAXEL_AREA   = GRID_STEP_MM ** 2   # 0.25 mm²

# 그리드 mm 좌표 (col, row → x, y)
_GRID_COORDS_MM = np.linspace(GRID_MIN_MM, GRID_MAX_MM, GRID_SIZE)   # [41]


# ─────────────────────────────────────────────────────────────────────────────
# SATSInferenceEngine
# ─────────────────────────────────────────────────────────────────────────────

class SATSInferenceEngine:
    """
    run_dir 에서 config.json + best_model.pt 를 로드하여 추론한다.

    Parameters
    ----------
    run_dir : str | Path   학습 run 디렉터리 경로
    device  : str          'cuda' | 'cpu' | 'auto'
    """

    def __init__(
        self,
        run_dir: str | Path,
        device: str = "auto",
        indenter_diameter_mm: float = 5.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        # ★체크포인트 **파일**을 직접 받을 수도 있다(runs/sats/<v*>/ 폴더 규약 —
        #   epoch_*.pt 등 best_model.pt 가 아닌 파일도 선택 가능). config.json 은
        #   같은 폴더에서 읽는다.
        ckpt_override: Path | None = None
        if self.run_dir.is_file():
            ckpt_override = self.run_dir
            self.run_dir = self.run_dir.parent
        self.indenter_diameter_mm = float(indenter_diameter_mm)

        # ── config 로드 ────────────────────────────────────────────────────
        cfg_path = self.run_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json 없음: {cfg_path}")

        raw = json.loads(cfg_path.read_text())
        valid = {f.name for f in fields(SATSConfig)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        self.cfg = SATSConfig(**filtered)

        # ── 디바이스 결정 ──────────────────────────────────────────────────
        if device == "auto":
            self.device = self.cfg.effective_device()
        else:
            self.device = device

        # ── 모델 로드 ──────────────────────────────────────────────────────
        ckpt_path = ckpt_override or self.run_dir / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

        self.model, self.ckpt_info = self._load_model(ckpt_path)
        self.window_size = self.cfg.window_size

        # 출력 grid 를 config 에서 읽는다 (0.5mm=41² 기본, 0.25mm=81², 0.1mm=201² 등).
        # 좌표·fz 적분·taxel 조회가 모두 이 값을 따르므로 fine 출력 모델도 정확.
        self.grid_size = int(self.cfg.grid_size)
        self.grid_min_mm = float(self.cfg.grid_min_mm)
        self.grid_max_mm = float(self.cfg.grid_max_mm)
        self.grid_step_mm = float(self.cfg.grid_step_mm)
        self.taxel_area = self.grid_step_mm ** 2
        self.grid_coords_mm = np.linspace(self.grid_min_mm, self.grid_max_mm, self.grid_size)

        # 크기입력(A) 모델이면 고정 지름을 조건으로 전달. 실시간에서는 접촉 크기를
        # 모르므로 디폴트 d5(학습 데이터 다수 클래스) — size 는 추론 대상이 아니라
        # GT magnitude 컨디셔닝이며, 위치 추정은 size 오지정에도 강건(peak corr 0.96).
        self.use_size_input = bool(getattr(self.cfg, "use_indenter_size_input", False))
        self._size_t = (
            torch.tensor([self.indenter_diameter_mm], dtype=torch.float32).to(self.device)
            if self.use_size_input else None
        )

        print(f"[InferenceEngine] 모델 로드 완료")
        print(f"  run_dir    : {self.run_dir}")
        print(f"  device     : {self.device}")
        print(f"  window_size: {self.window_size}")
        print(f"  output_grid: {self.grid_size}x{self.grid_size} @ {self.grid_step_mm:g}mm")
        print(f"  checkpoint : {self.ckpt_info['path']}")
        print(f"  ckpt_epoch : {self.ckpt_info['epoch']}")
        if self.use_size_input:
            print(f"  size_input : ON — indenter d={self.indenter_diameter_mm:g} mm 고정 조건")
        print(f"  strict_load: {self.ckpt_info['strict_load']}")
        print(f"  state_keys : {self.ckpt_info['n_state_tensors']}")
        print(f"  n_params   : {self.ckpt_info['n_model_params']}")

    # ── 공개 API ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        window : ndarray[window_size, 16]  (s_norm float32)

        Returns
        -------
        pred_map : ndarray[41, 41]  (압력 맵, 학습 스케일)
        """
        sensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
        # [1, window_size, 16]
        lengths = torch.tensor([self.window_size], dtype=torch.int64).to(self.device)

        if self.use_size_input:
            out = self.model(sensor, lengths, self._size_t)
        else:
            out = self.model(sensor, lengths)
        pred = out[0] if isinstance(out, tuple) else out
        return pred[0].cpu().numpy()   # [grid, grid]

    def get_peak(self, pred_map: np.ndarray) -> Tuple[float, float, float]:
        """
        예측 맵에서 peak 위치와 값을 반환한다 (config grid 좌표 기준).

        Returns
        -------
        (x_mm, y_mm, peak_value)
        """
        idx = np.unravel_index(np.argmax(pred_map), pred_map.shape)
        row, col = idx
        x_mm = float(self.grid_coords_mm[col])
        y_mm = float(self.grid_coords_mm[row])
        return x_mm, y_mm, float(pred_map[row, col])

    def get_fz(self, pred_map: np.ndarray) -> float:
        """
        Fz 추정 = sum(pred_map) × taxel_area [N]

        taxel_area 는 config grid_step² (해상도별 셀 면적) → 적분값은 해상도 불변.
        pred_map 단위는 학습 스케일(×100)이므로 /100 적용.
        """
        return float(pred_map.clip(0).sum()) * self.taxel_area / 100.0

    def set_diameter(self, diameter_mm: float) -> None:
        """실시간 인덴터 지름 변경 — size 조건(_size_t)·저장값 갱신(모델 재로드 없음).

        배포 모델은 size-conditioning(단일 모델)이라 d5/d10 전환이 조건 교체만으로 된다.
        위치/fz 게이트(min_distance·z_calib)는 호출측(extract_contacts)이 지름을 받아 처리.
        """
        self.indenter_diameter_mm = float(diameter_mm)
        if self.use_size_input:
            self._size_t = torch.tensor([self.indenter_diameter_mm],
                                        dtype=torch.float32).to(self.device)

    def get_taxel_value(self, pred_map: np.ndarray, x_mm: float, y_mm: float) -> float:
        """특정 (x_mm, y_mm) 위치의 taxel 값을 반환한다 (config grid 기준)."""
        col = int(round((x_mm - self.grid_min_mm) / self.grid_step_mm))
        row = int(round((y_mm - self.grid_min_mm) / self.grid_step_mm))
        col = max(0, min(self.grid_size - 1, col))
        row = max(0, min(self.grid_size - 1, row))
        return float(pred_map[row, col])

    # ── 내부 구현 ─────────────────────────────────────────────────────────────

    def _load_model(self, ckpt_path: Path) -> Tuple[torch.nn.Module, Dict[str, object]]:
        from sats.training.cnn_module import SATSCNNStage

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError(
                f"체크포인트 형식 오류: {ckpt_path} 에 'model' 키가 없습니다."
            )

        state_dict = ckpt["model"]
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"체크포인트 형식 오류: {ckpt_path} 의 'model' 값이 state_dict(dict)가 아닙니다."
            )

        model = SATSCNNStage(self.cfg)
        strict_loaded = True

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                raise RuntimeError(f"체크포인트 아키텍처 불일치:\n{e}") from None
            strict_loaded = False
            result = model.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                print(f"  [경고] 초기화되지 않은 키: {result.missing_keys}")
            if result.unexpected_keys:
                print(f"  [경고] 체크포인트에만 존재하는 키: {result.unexpected_keys}")

        model.eval()
        model = model.to(self.device)
        ckpt_info = {
            "path": str(ckpt_path),
            "epoch": ckpt.get("epoch", "unknown"),
            "strict_load": strict_loaded,
            "n_state_tensors": len(state_dict),
            "n_model_params": sum(p.numel() for p in model.parameters()),
        }
        return model, ckpt_info
