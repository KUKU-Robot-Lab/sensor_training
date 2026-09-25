"""격자 간격은 맵 크기에서 유도해야 한다 — 상수로 박으면 좌표가 배로 어긋난다."""
import numpy as np
import torch

from sats.bending.eval_contact_preservation import _grid_step_mm, _peak_xy


def _map_with_peak(width: int, row: int, col: int) -> torch.Tensor:
    m = torch.zeros(1, width, width)
    m[0, row, col] = 1.0
    return m


def test_step_matches_deploy_resolutions():
    assert _grid_step_mm(41) == 0.5      # g05
    assert _grid_step_mm(81) == 0.25     # g025 ← 실제 배포 run
    assert abs(_grid_step_mm(201) - 0.1) < 1e-12


def test_center_peak_maps_to_origin_at_every_resolution():
    for w in (41, 81, 201):
        xy = _peak_xy(_map_with_peak(w, w // 2, w // 2))
        assert np.allclose(xy, [[0.0, 0.0]]), f"{w}² 에서 중심이 원점이 아님: {xy}"


def test_corner_peak_maps_to_physical_bounds():
    for w in (41, 81, 201):
        assert np.allclose(_peak_xy(_map_with_peak(w, 0, 0)), [[-10.0, -10.0]])
        assert np.allclose(_peak_xy(_map_with_peak(w, w - 1, w - 1)), [[10.0, 10.0]])
