"""
SATS 학습 설정.

trial_id 명명 규칙: "{material}_d{D}_z{Z}_test{N}"
  - material : 예) ecomesh, ecomesh_xy0p5, eco20_xy1
  - D : 인덴터 직경 (mm), 예) 5, 10
  - Z : z_max (mm), 예) 1 → 1.0mm, 1.5 → 1.5mm
  - N : 반복 번호

파일 경로 매핑:
  raw BIN   : learning_data/sensor_raw_bin/ecomesh/d{D}/z_{Z_full}mm/test{N}/..._merged.bin
  raw CSV   : learning_data/sensor_raw_bin/ecomesh/d{D}/z_{Z_full}mm/test{N}/..._merged.csv (compat/export)
  baseline  : learning_data/sensor_raw_bin/ecomesh/d{D}/z_{Z_full}mm/test{N}/..._baseline.json
  GT npy    : learning_data/gt/{trial_id}_targets.npy
  GT index  : learning_data/gt/dataset_index.json

baseline JSON 키: Skin1_mean ~ Skin16_mean  (merged CSV 컬럼: s1 ~ s16)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ──────────────────────────────────────────────
# trial_id 파싱 / 경로 헬퍼
# ──────────────────────────────────────────────

def _parse_trial_id(trial_id: str) -> dict:
    """
    "ecomesh_d10_z1.5_test2" → {"material": "ecomesh", "d": 10, "z": 1.5, "n": 2}
    "ecomesh_xy0p5_d5_z2.5_test1" → {"material": "ecomesh_xy0p5", "d": 5, "z": 2.5, "n": 1}
    """
    m = re.fullmatch(r"(.+)_d(\d+)_z([0-9.]+)_test(\d+)", trial_id)
    if m is None:
        raise ValueError(f"trial_id 파싱 실패: {trial_id!r}")
    return {
        "material": m.group(1),
        "d": int(m.group(2)),
        "z": float(m.group(3)),
        "n": int(m.group(4)),
    }


def trial_id_to_paths(trial_id: str, raw_dir: str = "learning_data/sensor_raw_bin") -> dict:
    """
    trial_id → merged BIN/CSV / baseline JSON 경로 반환.

    Returns
    -------
    {
        "merged_bin" : Path,
        "merged_csv" : Path,
        "baseline_json" : Path,
    }
    """
    p = _parse_trial_id(trial_id)
    mat  = p["material"]                       # ecomesh
    d    = p["d"]                              # 10
    z    = p["z"]                              # 1.0
    n    = p["n"]                              # 1

    # z_full: 1.0 → "1.0", 1.5 → "1.5"
    z_full = f"{z:.1f}" if z == int(z) else f"{z}"

    base = Path(raw_dir) / mat / f"d{d}" / f"z_{z_full}mm" / f"test{n}"
    stem = f"{mat}_d{d}_z{z_full}_test{n}"

    return {
        "merged_bin":    base / f"{stem}_merged.bin",
        "merged_csv":    base / f"{stem}_merged.csv",
        "baseline_json": base / f"{stem}_baseline.json",
    }


def filter_trial_ids(
    trial_ids: List[str],
    *,
    include_materials: List[str] | None = None,
    exclude_diameters: List[int] | None = None,
) -> List[str]:
    """Filter trial ids by material key and indenter diameter while preserving order."""

    include_set = set(include_materials or [])
    exclude_set = set(exclude_diameters or [])
    filtered: List[str] = []
    for trial_id in trial_ids:
        parsed = _parse_trial_id(trial_id)
        if include_set and parsed["material"] not in include_set:
            continue
        if exclude_set and parsed["d"] in exclude_set:
            continue
        filtered.append(trial_id)
    return filtered


# ──────────────────────────────────────────────
# 메인 설정 dataclass
# ──────────────────────────────────────────────

@dataclass
class SATSConfig:
    """SATS 전체 학습 파이프라인 설정."""

    # ── 경로 ──────────────────────────────────────────────────────────────────
    raw_dir: str = "learning_data/sensor_raw_bin"
    gt_dir: str = "learning_data/gt"
    dataset_index_path: str = "learning_data/gt/dataset_index.json"
    out_dir: str = "sats/training/runs"
    run_name: str = "lstm_v1"

    # ── 소재 / trial 선택 ─────────────────────────────────────────────────────
    material: str = "ecomesh"
    include_materials: List[str] = field(default_factory=list)

    # validation에 사용할 trial_id 목록 (나머지는 train).
    # val_ratio > 0 이면 이 값은 무시되고 랜덤 sequence-level split이 사용된다.
    val_trials: List[str] = field(default_factory=list)

    # val_ratio > 0: 전체 시퀀스를 랜덤하게 (1-val_ratio) / val_ratio 로 분리 (논문 방식).
    # 0.0 이면 val_trials 기반 trial-level split 사용.
    val_ratio: float = 0.2

    # 학습/검증 풀에서 제외할 인덴터 직경(mm) 목록.
    # 예) [10] → d10 trial 전체 제거. val_trials에 포함된 d10 trial도 함께 제거됨.
    exclude_diameters: List[int] = field(default_factory=list)

    # ── 데이터 필터 (GT generation과 동일 기준) ───────────────────────────────
    grid_step_mm: float  = 0.5      # 그리드 간격
    grid_tol_mm: float   = 0.05     # on-grid 허용 오차
    grid_min_mm: float   = -10.0    # 그리드 범위 [min, max]
    grid_max_mm: float   = 10.0
    fz_min_abs_n: float  = 0.1      # |Fz| < 이 값이면 GT = 0 (영압 행)
    grid_size: int       = 41       # GT 맵 한 변 크기
    # u_mm은 node 내부 대기/가상 축이며 물리적 전단이나 depth 기준이 아니다.
    # mk555 기본 학습/GT는 u 필터를 적용하지 않고 z_stage_mm 기반 depth를 사용한다.
    use_u_zero_only: bool = False   # False=전체 row 포함
    u_zero_tol_mm: float = 1e-6     # use_u_zero_only=True일 때만 의미
    prefer_merged_bin: bool = True  # 학습 입력 기본값: merged BIN 우선, CSV는 fallback

    # ── GT 생성 방식 ────────────────────────────────────────────────────────
    # "precomputed": 기존 learning_data/gt/*_targets.npy 사용.
    # "on_the_fly": merged BIN의 x/y/z_depth/Fz로 구형 인덴터 GT를 CPU worker에서 즉석 생성.
    # "gpu_on_the_fly": DataLoader는 compact metadata만 반환하고 train step에서 GPU batch GT 생성.
    gt_mode: str = "precomputed"
    gt_scale: float = 100.0
    gt_meta_cache_dir: str = "learning_data/gt_meta_cache"
    use_gt_meta_cache: bool = True
    on_the_fly_z_s_mm: float = 2.0
    on_the_fly_patch_step_mm: float = 0.1
    contact_radius_step_mm: float = 0.05
    min_contact_radius_mm: float = 0.05
    z_depth_min_mm: float = 0.001
    # (B) β_geom force conservation: rescale base kernel so ∫∫(σ_zz/F) dA = 1
    # (Boussinesq equilibrium). Corrects patch-discretization over/under-conservation.
    # Near no-op for radii ≥1mm; matters for sub-patch-step radii / fine grids.
    gt_beta_force_conservation: bool = False
    # Indenter-size conditioning: feed the known indenter diameter as a FiLM
    # condition so the model can produce size-appropriate pressure peaks for
    # mixed d5/d10 data (resolves the size-blend d10 low-force over-prediction).
    # Diameter is a known probe property; radius/depth are NOT (unknown at inference).
    use_indenter_size_input: bool = False
    # (C) β(p) material rectification (paper Note S3/S9): β(p)=c0+c1·p+c2·p²,
    # p = mean contact pressure in kPa = |fz|/(π·a²)·1000. Ported from the
    # precomputed generate_gt.compute_beta so on-the-fly GT matches. Coefficients
    # are material-specific (hyperelastic stiffening) and must be calibrated.
    gt_beta_mode: str = "none"          # "none" | "poly2"
    gt_beta_c0: float = 1.0
    gt_beta_c1: float = 0.0
    gt_beta_c2: float = 0.0
    gt_beta_min: float = 0.2
    gt_beta_max: float = 5.0
    # Mixed precision (bf16 autocast): ~2x speed + ~half activation memory.
    # Needed for very fine output grids (0.25/0.1mm). bf16 needs no GradScaler.
    use_amp: bool = False

    # on-the-fly GT window sampling.
    # balanced_contact는 loading/dynamic, plateau/static, saturation, z/Fz bin
    # 대표 샘플을 함께 보존해 U 정지 시간 차이로 인한 과대표집을 줄인다.
    on_the_fly_sampling_policy: str = "balanced_contact"
    loading_stride: int = 1
    plateau_stride: int = 10
    saturation_stride: int = 2
    saturation_fz_frac: float = 0.9
    saturation_z_frac: float = 0.9
    fz_balance_bins: int = 10
    z_balance_bins: int = 6
    z_balance_bin_width_mm: float = 0.005

    # ── 시퀀스 구성 ───────────────────────────────────────────────────────────
    # 한 (x_mm, y_mm) 위치의 압입 사이클 = 하나의 원본 시퀀스.
    # 논문식 window 학습에서도 먼저 이 길이까지 cycle 앞부분을 보존한 뒤,
    # loading phase에서 window_size=10 슬라이딩 윈도우를 만든다.
    seq_len: int     = 1000     # max timestep (mk555 d5 peak: ~820-860)
    min_seq_len: int = 3        # 이보다 짧은 시퀀스는 학습에서 제외

    # ── 센서 ──────────────────────────────────────────────────────────────────
    n_sensors: int = 16
    # merged CSV 컬럼명
    sensor_cols: List[str] = field(
        default_factory=lambda: [f"s{i}" for i in range(1, 17)]
    )
    # baseline JSON 키 (Skin1_mean ~ Skin16_mean)
    baseline_mean_keys: List[str] = field(
        default_factory=lambda: [f"Skin{i}_mean" for i in range(1, 17)]
    )

    # ── LSTM 하이퍼파라미터 ───────────────────────────────────────────────────
    hidden_dim: int    = 64      # LSTM hidden size (per sensor)
    num_layers: int    = 2       # LSTM 깊이
    dropout: float     = 0.1    # LSTM 내부 dropout (num_layers > 1 일 때 유효)
    bidirectional: bool = False  # 단방향: 실시간 추론 호환

    # ── Self-Attention 하이퍼파라미터 ─────────────────────────────────────────
    attn_dim: int      = 64      # GAT 선형 투영 차원 (논문: 125)
    n_gat_layers: int  = 2       # GAT 레이어 수 (논문: 2)

    # ── Local Map 하이퍼파라미터 ──────────────────────────────────────────────
    local_map_size: int       = 15    # 각 센서의 local map 한 변 크기 (홀수 권장)
    sensor_spacing_mm: float  = 6.5   # 센서 간 물리 간격 (mm)

    # ── CNN 하이퍼파라미터 ────────────────────────────────────────────────────
    cnn_hidden_channels: int = 16  # CNN Refiner 중간 채널 수 (논문 미명시 → 기본 16)

    # ── Ablation 플래그 (논문 Table S2 / FigS19) ──────────────────────────────
    ablate_lstm: bool = False       # LSTM → 비순환 통계 인코더(mean/max/last)로 대체
    ablate_attention: bool = False  # self-attention 출력을 0으로 (이웃 정보 집계 제거)
    ablate_cnn: bool = False        # CNN refiner 생략 (merged_map 을 최종 출력으로)

    # ── 체크포인트 연계 ───────────────────────────────────────────────────────
    lstm_ckpt: str      = ""     # 사전학습된 LSTM 체크포인트 경로 (빈 문자열=미사용)
    attn_ckpt: str      = ""     # 사전학습된 Self-Attention 체크포인트 경로
    local_map_ckpt: str = ""     # 사전학습된 Local Map 체크포인트 경로

    # ── 학습 ──────────────────────────────────────────────────────────────────
    # 논문 기준: batch_size=2048, lr=0.0064 (고정, 스케줄러 없음).
    # 현재 RTX 4090 + window_size=10 경로에서는 2048이 VRAM/RAM 균형이 좋다.
    batch_size: int  = 2048
    lr: float        = 1e-3
    weight_decay: float = 1e-5
    epochs: int      = 50
    lr_patience: int = 5        # ReduceLROnPlateau patience
    lr_factor: float = 0.5
    clip_grad: float = 1.0      # gradient clipping (None 이면 비활성)
    use_lr_scheduler: bool = True  # False → 고정 LR (논문 방식)
    num_workers: int = 2
    dataloader_prefetch_factor: int = 4
    persistent_workers: bool = True
    seed: int        = 42
    save_every: int  = 10       # N epoch마다 체크포인트 저장

    # ── 윈도우 데이터셋 설정 ──────────────────────────────────────────────────
    # 논문 방식: loading phase만, window_size=10 슬라이딩 윈도우.
    # 각 학습 샘플은 sensor_window [10,16] → 마지막 timestep의 GT map [41,41].
    window_size: int = 10
    use_window_dataset: bool = True

    # ── 계산 장치 ─────────────────────────────────────────────────────────────
    device: str = "cuda"        # "cuda" | "cpu"

    # ─────────────────────────────────────────────────────────────────────────
    # 경로 유틸리티 (dataclass 메서드)
    # ─────────────────────────────────────────────────────────────────────────

    def run_dir(self) -> Path:
        """학습 결과 저장 디렉터리."""
        return Path(self.out_dir) / self.run_name

    def gt_npy_path(self, trial_id: str) -> Path:
        return Path(self.gt_dir) / f"{trial_id}_targets.npy"

    def trial_paths(self, trial_id: str) -> dict:
        """trial_id → merged_bin / merged_csv / baseline_json 경로."""
        return trial_id_to_paths(trial_id, raw_dir=self.raw_dir)

    def effective_device(self) -> str:
        """cuda 요청이지만 불가능하면 cpu로 폴백."""
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device
