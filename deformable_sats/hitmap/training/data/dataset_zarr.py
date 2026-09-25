"""
dataset_zarr.py

Zarr 포맷 데이터셋 로더. GPU 학습 효율을 위해 최적화됨.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from training.data.sensor_layout import DEAD_CHANNEL_INDICES

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False


def _same_path(left: Union[str, Path], right: Union[str, Path]) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _index_candidates(zarr_path: Path) -> List[Path]:
    return [
        zarr_path / "dataset_index.json",
        zarr_path.parent / f"{zarr_path.stem}_index.json",
        zarr_path.parent / "dataset_index.json",
    ]


def _compact_index_path(zarr_path: Path) -> Path:
    return zarr_path / "dataset_index_compact.npz"


def _load_zarr_index(zarr_path: Path, zg) -> List[Dict]:
    index_path = next((p for p in _index_candidates(zarr_path) if p.exists()), None)
    if index_path is None:
        searched = ", ".join(str(p) for p in _index_candidates(zarr_path))
        raise FileNotFoundError(f"No dataset index found for {zarr_path}. Searched: {searched}")

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    all_samples = index_data.get("samples", [])
    if not isinstance(all_samples, list):
        raise ValueError(f"Invalid dataset index format: {index_path}")

    for sample in all_samples:
        sample_zarr = sample.get("zarr_path")
        if sample_zarr and not _same_path(sample_zarr, zarr_path):
            raise ValueError(
                f"Index {index_path} references a different zarr: {sample_zarr} != {zarr_path}"
            )
        zarr_index = int(sample["zarr_index"])
        n_rows = int(zg["tactile_lr_norm"].shape[0])
        if zarr_index < 0 or zarr_index >= n_rows:
            raise ValueError(
                f"Index {index_path} has zarr_index={zarr_index} outside {zarr_path} row count {n_rows}"
            )
    return all_samples


def _extract_scalar_json_value(line: str):
    _, raw = line.split(":", 1)
    return json.loads(raw.strip().rstrip(","))


def _build_compact_index_from_json(zarr_path: Path, zg) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n_rows = int(zg["tactile_lr_norm"].shape[0])
    trial_codes = np.full(n_rows, -1, dtype=np.int32)
    phase_codes = np.full(n_rows, 255, dtype=np.uint8)
    trial_vocab: list[str] = []
    trial_to_code: dict[str, int] = {}
    assigned = 0
    index_path = next((p for p in _index_candidates(zarr_path) if p.exists()), None)
    if index_path is None:
        searched = ", ".join(str(p) for p in _index_candidates(zarr_path))
        raise FileNotFoundError(f"No dataset index found for {zarr_path}. Searched: {searched}")

    print(f"  [all] compact index 생성 중... ({index_path})", flush=True)
    all_samples = _load_zarr_index(zarr_path, zg)
    for sample in all_samples:
        current_trial = str(sample["trial_id"])
        current_phase = str(sample.get("phase", "all"))
        zarr_index = int(sample["zarr_index"])
        code = trial_to_code.get(current_trial)
        if code is None:
            code = len(trial_vocab)
            trial_to_code[current_trial] = code
            trial_vocab.append(current_trial)
        trial_codes[zarr_index] = code
        phase_codes[zarr_index] = 0 if current_phase == "loading" else 1 if current_phase == "unloading" else 2
        assigned += 1
        if assigned % 500000 == 0:
            print(f"  [all] compact index 진행: {assigned:,}/{n_rows:,}", flush=True)

    if assigned != n_rows:
        raise ValueError(f"Compact index assignment mismatch: assigned {assigned:,} rows, expected {n_rows:,}")
    cache_path = _compact_index_path(zarr_path)
    np.savez(cache_path, trial_codes=trial_codes, phase_codes=phase_codes, trial_vocab=np.array(trial_vocab, dtype=object))
    print(f"  [all] compact index 저장 완료: {cache_path}", flush=True)
    return trial_codes, phase_codes, trial_vocab


def _load_compact_index(zarr_path: Path, zg) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cache_path = _compact_index_path(zarr_path)
    if cache_path.exists():
        index_path = next((p for p in _index_candidates(zarr_path) if p.exists()), None)
        if index_path is not None and cache_path.stat().st_mtime < index_path.stat().st_mtime:
            return _build_compact_index_from_json(zarr_path, zg)
        print(f"  [all] compact index 로드 중... ({cache_path})", flush=True)
        cache = np.load(cache_path, allow_pickle=True)
        trial_codes = cache["trial_codes"].astype(np.int32, copy=False)
        phase_codes = cache["phase_codes"].astype(np.uint8, copy=False)
        trial_vocab = [str(x) for x in cache["trial_vocab"].tolist()]
        n_rows = int(zg["tactile_lr_norm"].shape[0])
        if trial_codes.shape[0] != n_rows or phase_codes.shape[0] != n_rows:
            return _build_compact_index_from_json(zarr_path, zg)
        print(f"  [all] compact index 로드 완료. ({trial_codes.shape[0]:,} rows)", flush=True)
        return trial_codes, phase_codes, trial_vocab
    return _build_compact_index_from_json(zarr_path, zg)


def _resolve_depth_array(zg, indices: List[int]) -> torch.Tensor:
    if "z_contact_mm" in zg:
        return torch.from_numpy(np.array(zg["z_contact_mm"].oindex[indices])).float()
    return torch.from_numpy(np.array(zg["depth_mm"].oindex[indices])).float()


class ZarrDataset(Dataset):
    """
    preprocess.py에서 생성한 .zarr 데이터를 로드하는 PyTorch Dataset.
    aux_feat 마지막 컬럼이 직경(diameter_mm) 또는 접촉 반경(contact_radius_mm)일 수 있으며
    zarr attrs["aux_last_field"] 값으로 구분한다.
    깊이값은 z_contact_mm가 있으면 그것을 우선 사용하고, 없으면 depth_mm로 fallback한다.
    
    Args:
        zarr_path: .zarr 디렉토리 경로
        split: "train", "val", "test", 또는 "all"
        phase: "loading", "unloading", 또는 "all"
        val_ratio: train/val 분할 비율
        seed: 난수 시드 (Trial 분할용)
    """

    def __init__(
        self,
        zarr_path: Union[str, Path],
        split: str = "all",
        phase: str = "loading",
        val_ratio: float = 0.2,
        seed: int = 42,
        drop_dead_channels: bool = False,
    ):
        if not ZARR_AVAILABLE:
            raise ImportError("zarr 라이브러리가 필요합니다. 'pip install zarr'를 실행하세요.")

        self.zarr_path = Path(zarr_path)
        self.split = split
        self.phase_filter = phase
        self.drop_dead_channels = drop_dead_channels
        
        # Zarr 데이터 열기
        zg = zarr.open_group(str(self.zarr_path), mode='r')
        self.aux_last_field = zg.attrs.get("aux_last_field", "diameter_mm")
        self.depth_source = str(zg.attrs.get("depth_source", "depth_mm"))
        self.target_depth_source = str(zg.attrs.get("target_depth_source", "z_contact_mm" if "z_contact_mm" in zg else "depth_mm"))
        self.aux_depth_source = str(zg.attrs.get("aux_depth_source", self.depth_source))
        self.stage_depth_source = str(zg.attrs.get("stage_depth_source", "z_stage_mm" if "z_stage_mm" in zg else self.depth_source))

        trial_codes, phase_codes, trial_vocab = _load_compact_index(self.zarr_path, zg)
        n_rows = int(trial_codes.shape[0])
        row_indices = np.arange(n_rows, dtype=np.int64)

        if phase != "all":
            wanted_phase = 0 if phase == "loading" else 1 if phase == "unloading" else 2
            phase_mask = phase_codes == wanted_phase
            row_indices = row_indices[phase_mask]
            print(f"  [{split}] phase='{phase}' 필터 적용 후 {row_indices.shape[0]:,} rows", flush=True)

        # 2. Trial 단위 분할
        available_trial_codes = np.unique(trial_codes[row_indices])
        trial_ids = sorted(trial_vocab[int(code)] for code in available_trial_codes if int(code) >= 0)
        rng = np.random.default_rng(seed)
        rng.shuffle(trial_ids)
        
        n_val = int(len(trial_ids) * val_ratio)
        if n_val == 0 and len(trial_ids) > 1:
            n_val = 1
            
        val_trials = set(trial_ids[:n_val])
        train_trials = set(trial_ids[n_val:])
        
        if split == "train":
            keep_trials = train_trials
        elif split == "val":
            keep_trials = val_trials
        elif split == "all":
            keep_trials = set(trial_ids)
        else:
            keep_trials = val_trials

        keep_mask = np.isin(trial_codes[row_indices], np.array([trial_vocab.index(t) for t in keep_trials], dtype=np.int32))
        selected_indices = row_indices[keep_mask]
        
        # 3. 데이터 메모리 로드 (병목 해결 핵심)
        print(f"  [{split}] 데이터를 메모리에 로드 중... ({selected_indices.shape[0]:,} 샘플)", flush=True)
        indices = selected_indices.tolist()
        
        # 필요한 컬럼만 oindex를 사용하여 한 번에 NumPy로 로드
        self.tactile_data = torch.from_numpy(np.array(zg["tactile_lr_norm"].oindex[indices])).float()
        if self.drop_dead_channels:
            live_idx = [i for i in range(self.tactile_data.shape[1]) if i not in DEAD_CHANNEL_INDICES]
            self.tactile_data = self.tactile_data[:, live_idx]
        self.aux_data = torch.from_numpy(np.array(zg["aux_feat"].oindex[indices])).float()
        self.cx_data = torch.from_numpy(np.array(zg["cx"].oindex[indices])).float()
        self.cy_data = torch.from_numpy(np.array(zg["cy"].oindex[indices])).float()
        self.depth_data = _resolve_depth_array(zg, indices)
        self.fz_data = torch.from_numpy(np.array(zg["fz"].oindex[indices])).float()

        self.indices = selected_indices
        self.samples = indices
        self.trial_ids = [trial_vocab[int(trial_codes[i])] for i in selected_indices]
        n_ch = self.tactile_data.shape[1]
        if self.drop_dead_channels:
            print(f"  [{split}] 로드 완료. (dead channel 제거 적용: tactile {n_ch}ch)", flush=True)
        else:
            print(f"  [{split}] 로드 완료. (tactile {n_ch}ch)", flush=True)
        
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.aux_last_field == "contact_radius_mm":
            radius_val = self.aux_data[idx, 3]
        else:
            radius_val = self.aux_data[idx, 3] / 2.0  # diameter → radius
        return {
            "tactile": self.tactile_data[idx],
            "radius":  radius_val,
            "target_sr": torch.stack([self.cx_data[idx], self.cy_data[idx], self.depth_data[idx]]),
            "target_fz": self.fz_data[idx:idx+1],
            "trial_id": self.trial_ids[idx]
        }
