"""밴딩 기하: 스테이지 압축량 δ → 곡률·각도(원호 좌굴 모델) + 센서 자가추정 κ̂.

취득 기구: 센서 양단 고정, 스테이지로 Y를 밀어 span을 δ만큼 압축 → 원호(arc) 좌굴.
호길이 L(=밀기 전 두 클램프 자유 길이)은 고정, 현(chord) c = L − δ.

원호 관계:  c/L = sinc(θ/2) = sin(θ/2)/(θ/2)   (θ = 전체 대응각[rad])
곡률:       κ = θ / L  [1/mm],  R = 1/κ,  bend angle = θ (또는 끝단 기울기 θ/2)

자가추정(논문 §5.3): 무접촉 밴딩 압력변화 Δp_i ≈ k_i·κ·z_i (z_i = taxel 중심축 거리).
등가중 최소자승 → κ̂ ∝ Σ_i z_i·Δp_i / Σ_i z_i²  (스케일 무관, 기하 GT로 캘리브레이션).

주의: 양단 완전고정(clamped)이면 엄밀 좌굴형상은 원호가 아니라 중앙 곡률 최대인
모드형상 → 원호 근사는 센서가 중앙부일 때 국소 성립. κ̂-vs-κ_geo 회귀로 유효범위 확인.
"""
from __future__ import annotations

import numpy as np

# 16채널 taxel (x, y) 좌표 [mm] — 4×4, pitch 6.5mm, ±9.75/±3.25.
# 밴딩축이 Y압축이므로 중심축 거리 z_i = 각 채널의 y좌표.
# (출처: visualizing_scripts/CenterLine, ch14 오타 -3.75→-3.25 교정)
TAXEL_XY_MM: dict[int, tuple[float, float]] = {
    1: (-9.75, -9.75), 2: (-3.25, -9.75), 3: (3.25, -9.75), 4: (9.75, -9.75),
    5: (-9.75, -3.25), 6: (-3.25, -3.25), 7: (3.25, -3.25), 8: (9.75, -3.25),
    9: (-9.75, 3.25), 10: (-3.25, 3.25), 11: (3.25, 3.25), 12: (9.75, 3.25),
    13: (-9.75, 9.75), 14: (-3.25, 9.75), 15: (3.25, 9.75), 16: (9.75, 9.75),
}
# 채널 순서(Skin1..16)대로의 중심축 거리 z_i [16]
TAXEL_Z_MM: np.ndarray = np.array([TAXEL_XY_MM[i][1] for i in range(1, 17)], dtype=float)

DEFAULT_BEND_LENGTH_MM = 35.0  # 유효 굽힘 길이 L (밀기 전 클램프 자유 길이)


def _sinc(x: np.ndarray) -> np.ndarray:
    """정규화 안 된 sinc = sin(x)/x, x→0 에서 1 (0 나눗셈 안전)."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def arc_angle_from_compression(delta_mm: np.ndarray, length_mm: float = DEFAULT_BEND_LENGTH_MM) -> np.ndarray:
    """압축량 δ[mm] → 원호 전체각 θ[rad]. (L−δ)/L = sinc(θ/2) 를 θ/2∈(0,π)에서 이분법.

    δ<0 (신장) 또는 δ≥L 은 물리적 arc 범위를 벗어나므로 각각 0, NaN 처리.
    sinc 은 (0,π)에서 1→0 단조감소이므로 해가 유일.
    """
    delta = np.atleast_1d(np.asarray(delta_mm, dtype=float))
    L = float(length_mm)
    if L <= 0:
        raise ValueError(f"length_mm must be > 0, got {L}")
    theta = np.zeros_like(delta)
    ratio = (L - delta) / L  # = c/L
    for k, r in enumerate(ratio):
        if not np.isfinite(r) or r >= 1.0:   # δ≤0 → 평평
            theta[k] = 0.0
            continue
        if r <= 0.0:                          # δ≥L → arc 무효
            theta[k] = np.nan
            continue
        lo, hi = 1e-9, np.pi                  # θ/2 ∈ (0, π), sinc 단조감소
        mid = 0.5 * (lo + hi)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if _sinc(np.array([mid]))[0] > r:
                lo = mid
            else:
                hi = mid
        theta[k] = 2.0 * mid
    return theta


def curvature_from_compression(delta_mm: np.ndarray, length_mm: float = DEFAULT_BEND_LENGTH_MM) -> dict[str, np.ndarray]:
    """압축량 δ → {theta_rad, theta_deg, kappa(1/mm), radius_mm}. 원호 GT.

    kappa = θ/L, radius = 1/kappa (θ=0 이면 inf). bend angle 은 θ(전체각).
    """
    theta = arc_angle_from_compression(delta_mm, length_mm)
    kappa = theta / float(length_mm)
    with np.errstate(divide="ignore"):
        radius = np.where(kappa > 0, 1.0 / kappa, np.inf)
    return {
        "theta_rad": theta,
        "theta_deg": np.degrees(theta),
        "kappa_per_mm": kappa,
        "radius_mm": radius,
    }


def self_estimated_curvature(delta_p: np.ndarray, z_mm: np.ndarray = TAXEL_Z_MM) -> np.ndarray:
    """센서 자가추정 곡률 κ̂ ∝ Σ z_i·Δp_i / Σ z_i²  (논문 §5.3, 스케일 무관).

    delta_p: [N,16] 채널별 baseline 대비 변화(부호). z_mm: [16] 중심축 거리.
    반환: [N] signed κ̂ (임의 스케일 — 기하 GT로 선형 캘리브레이션 필요).
    """
    dp = np.atleast_2d(np.asarray(delta_p, dtype=float))
    z = np.asarray(z_mm, dtype=float)
    if dp.shape[1] != z.shape[0]:
        raise ValueError(f"delta_p last dim {dp.shape[1]} != z dim {z.shape[0]}")
    denom = float(np.sum(z ** 2))
    if denom <= 0:
        raise ValueError("Σz² must be > 0")
    return (dp @ z) / denom


def calibrate(k_hat: np.ndarray, kappa_geo: np.ndarray) -> dict[str, float]:
    """κ̂(임의스케일) ↔ κ_geo(GT) 선형회귀. slope·intercept·R² 반환 (교차검증)."""
    a = np.asarray(k_hat, dtype=float)
    b = np.asarray(kappa_geo, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": int(a.size)}
    slope, intercept = np.polyfit(a, b, 1)
    pred = slope * a + intercept
    ss_res = float(np.sum((b - pred) ** 2))
    ss_tot = float(np.sum((b - np.mean(b)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "n": int(a.size)}
