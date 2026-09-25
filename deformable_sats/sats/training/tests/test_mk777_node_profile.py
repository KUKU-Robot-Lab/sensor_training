"""mk555 EtherMotion node contract tests."""
from pathlib import Path

import pytest

from sats.preprocessing.node_profile import parse_node_profile


ROOT = Path(__file__).resolve().parents[3]
NODE_DIR = ROOT / "skin_ws" / "node"


@pytest.mark.parametrize(
    ("filename", "z_min", "z_max", "depth", "u_max"),
    [
        ("SATS_d5_mk555.node", 13.00, 15.50, 2.50, 3.0),
        ("SATS_d10_mk555.node", 12.00, 15.50, 3.50, 4.0),
    ],
)
def test_mk555_node_grid_and_z_profile(filename, z_min, z_max, depth, u_max):
    node_path = NODE_DIR / filename
    if not node_path.exists():
        # 대용량 취득 .node 데이터는 저장소에서 제외됨(gitignore). 로컬에 있을 때만 검증.
        pytest.skip(f"{filename} 미존재 (gitignore된 대용량 취득 데이터)")

    profile = parse_node_profile(node_path)

    assert profile.grid_size_x == 41
    assert profile.grid_size_y == 41
    assert profile.xy_count == 1681
    assert profile.x_min_mm == pytest.approx(-10.0)
    assert profile.x_max_mm == pytest.approx(10.0)
    assert profile.y_min_mm == pytest.approx(-10.0)
    assert profile.y_max_mm == pytest.approx(10.0)
    assert profile.xy_step_mm == pytest.approx(0.5)

    assert profile.z_min_mm == pytest.approx(z_min)
    assert profile.z_max_mm == pytest.approx(z_max)
    assert profile.z_step_mm == pytest.approx(0.5)
    assert profile.z_depth_mm == pytest.approx(depth)
    assert profile.u_values_mm[0] == pytest.approx(0.0)
    assert profile.u_values_mm[-1] == pytest.approx(u_max)
    assert profile.first_u_cycle_mm == (0.0, 0.5, 0.0)
