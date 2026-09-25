"""실시간 데모: SATS 압력맵 → 접촉별 (x,y,z,fz) 추출 + 터미널 포매팅 + 최적프레임 latch.

단일/다중(최대 3) 공통. z 는 z_calibration(A1) LUT로 근사, fz 는 접촉별 Voronoi 적분.
좌표/면적은 엔진 grid(config) 기준(0.5/0.25/0.1mm 출력 모델 모두 지원).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from sats.tools.multicontact_metrics import detect_peaks


@dataclass(frozen=True)
class Contact:
    """단일 접촉 추정치."""
    x_mm: float
    y_mm: float
    z_mm: float
    fz_n: float
    peak_val: float


# ─────────────────────────────────────────────────────────────────────────────
# 상태 판정(무접촉/무하중 vs 접촉) — 3모드(contacts·theta·bending) 공통 단일 소스
# ─────────────────────────────────────────────────────────────────────────────

class ContactState(Enum):
    """센서 채널의 표시 상태."""
    OFFLINE = "offline"          # 센서 미연결/데이터 없음
    NO_CONTACT = "no_contact"    # 연결됨·무접촉(무하중)
    CONTACT = "contact"          # 접촉 감지
    BENDING = "bending"          # 밴딩 중·무접촉(bending 모드 전용)


# 배너 색 — 초록 heatmap/포스터 대비 고대비
STATE_COLORS: dict[ContactState, str] = {
    ContactState.OFFLINE: "#bbbbbb",
    ContactState.NO_CONTACT: "#888888",
    ContactState.CONTACT: "#d81e00",
    ContactState.BENDING: "#0033cc",
}


@dataclass(frozen=True)
class StateBanner:
    """상태 배너 표시 payload — 라벨·색·활성(접촉有) 플래그."""
    state: ContactState
    label: str
    color: str

    @property
    def active(self) -> bool:
        return self.state is ContactState.CONTACT


def state_banner(contacts: list[Contact] | None, *, theta_deg: float | None = None,
                 connected: bool = True, theta_band_deg: float = 20.0) -> StateBanner:
    """접촉 리스트(+선택 theta)로 표시 상태 판정. 3모드 공통 게이트.

    - connected=False           → OFFLINE
    - 접촉 있음                  → CONTACT (theta 있으면 라벨에 병기)
    - 접촉 없음·theta≥band       → BENDING(밴딩 중 무접촉)
    - 접촉 없음·그 외            → NO_CONTACT
    무접촉 판정 자체는 extract_contacts 의 fz/peak 게이트가 이미 수행하므로,
    여기서는 그 결과를 일관된 배너로 변환만 한다(표시 통일).
    """
    if not connected:
        return StateBanner(ContactState.OFFLINE, "SENSOR OFFLINE",
                           STATE_COLORS[ContactState.OFFLINE])
    n = len(contacts) if contacts else 0
    if n > 0:
        label = "CONTACT" if n == 1 else f"CONTACT x{n}"
        if theta_deg is not None:
            label += f"   theta {theta_deg:+.0f}°"
        return StateBanner(ContactState.CONTACT, label, STATE_COLORS[ContactState.CONTACT])
    if theta_deg is not None and abs(theta_deg) >= theta_band_deg:
        return StateBanner(ContactState.BENDING, f"BENDING {theta_deg:+.0f}° (no contact)",
                           STATE_COLORS[ContactState.BENDING])
    return StateBanner(ContactState.NO_CONTACT, "NO CONTACT",
                       STATE_COLORS[ContactState.NO_CONTACT])


def apply_position_gain(x_mm: float, y_mm: float, gain: float,
                        center_mm: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """과대응답(gain>1) 보정 — 중심 기준 반경을 1/gain 로 축소.

    실측(D10, v6): 모터 1mm 이동에 예측이 1.8mm 이동(gain 1.80) = 위치 과장.
    p = c + g·(s − c) 모델이므로 역보정은 s = c + (p − c)/g.
    ⚠ 'center 에서는 정확하다'는 가정 — 1D 3mm 구간에서만 측정했으므로 전면(±10mm)
    검증 안 됨. 기본 gain=1.0(off), 필요 시 --position-gain 으로 활성화.
    """
    if gain == 1.0 or gain <= 0:
        return x_mm, y_mm
    cx, cy = center_mm
    return cx + (x_mm - cx) / gain, cy + (y_mm - cy) / gain


def extract_contacts(pred_map: np.ndarray, *, grid_min_mm: float, grid_step_mm: float,
                     taxel_area: float, diameter_mm: float, max_contacts: int = 3,
                     min_distance_mm: float = 3.0, rel_threshold: float = 0.3,
                     min_fz: float = 0.0, min_peak_val: float = 0.0,
                     position_gain: float = 1.0,
                     z_calib=None) -> list[Contact]:
    """압력맵 → 접촉 리스트(총 fz 내림차순). z_calib 없으면 z=nan.

    fz: 양압 셀을 최근접 peak에 Voronoi 배정 후 셀합×area/100 (총합 보존, 다접촉 분할).
    ★무접촉 게이트: peak_val<min_peak_val 프레임은 빈 리스트, 접촉별 fz<min_fz 는 제거.
    (detect_peaks 는 상대임계라 무접촉 노이즈도 전역최대를 잡음 → 절대 하한 필요.)
    """
    if float(np.clip(pred_map, 0, None).max()) < min_peak_val:
        return []
    peaks = detect_peaks(pred_map, grid_min_mm=grid_min_mm, grid_step_mm=grid_step_mm,
                         min_distance_mm=min_distance_mm, rel_threshold=rel_threshold,
                         max_peaks=max_contacts, subpixel=True)   # 서브픽셀(< grid_step) 판독
    if len(peaks) == 0:
        return []
    h, w = pred_map.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cell_x = grid_min_mm + cc * grid_step_mm
    cell_y = grid_min_mm + rr * grid_step_mm
    pos = np.clip(pred_map, 0, None)
    dists = np.stack([np.hypot(cell_x - px, cell_y - py) for px, py, _ in peaks], axis=0)  # [K,H,W]
    assign = dists.argmin(0)
    contacts: list[Contact] = []
    for k, (px, py, pv) in enumerate(peaks):
        fz = float(pos[assign == k].sum()) * taxel_area / 100.0
        if fz < min_fz:                                  # 약한 스퓨리어스(무접촉) 제거
            continue
        z = float(z_calib.z_from_peak(pv, diameter_mm)) if z_calib is not None else float("nan")
        gx, gy = apply_position_gain(float(px), float(py), position_gain)   # 과대응답 보정
        contacts.append(Contact(x_mm=gx, y_mm=gy, z_mm=z, fz_n=fz, peak_val=float(pv)))
    return sorted(contacts, key=lambda c: -c.fz_n)


def format_contacts(frame_idx: int, contacts: list[Contact], theta_deg: float | None = None) -> str:
    """터미널 한 줄(들). theta 지정 시 헤더에 표시."""
    head = f"[frame {frame_idx:6d}]"
    if theta_deg is not None:
        head += f"  theta={theta_deg:+6.1f} deg"
    head += f"  {len(contacts)} contact(s)"
    lines = [head]
    for i, c in enumerate(contacts, 1):
        z = "  n/a" if np.isnan(c.z_mm) else f"{c.z_mm:4.2f}mm"
        lines.append(f"    #{i}  x={c.x_mm:+6.1f}  y={c.y_mm:+6.1f}  z={z}  fz={c.fz_n:6.2f}N")
    return "\n".join(lines)


def format_contacts_line(contacts: list[Contact], theta_deg: float | None = None, width: int = 118) -> str:
    """한 줄 갱신용 컴팩트 포맷(\\r 로 덮어쓰기). 무접촉이면 'no contact'."""
    head = "" if theta_deg is None else f"theta={theta_deg:+5.1f}  "
    if not contacts:
        body = "no contact"
    else:
        parts = []
        for i, c in enumerate(contacts, 1):
            z = "n/a" if np.isnan(c.z_mm) else f"{c.z_mm:.2f}"
            parts.append(f"#{i} ({c.x_mm:+.1f},{c.y_mm:+.1f}) z{z} fz{c.fz_n:.2f}N")
        body = "  ".join(parts)
    return (head + f"[{len(contacts)}] " + body).ljust(width)[:width]


def format_contacts_block(contacts: list[Contact], *, theta_deg: float | None = None,
                          frame: int = 0, fps: float = 0.0, mode: str = "contacts") -> list[str]:
    """nvidia-smi 식 제자리 갱신용 블록(라인 리스트). 최대 3접촉 + 헤더."""
    lines = [f" v6 SATS realtime — {mode}    frame {frame:<7d} {fps:5.1f} fps"]
    if theta_deg is not None:
        lines.append(f" bending theta : {theta_deg:+6.1f} deg")
    lines.append(" " + "-" * 52)
    if not contacts:
        lines.append("  (no contact)")
    else:
        for i, c in enumerate(contacts, 1):
            z = " n/a " if np.isnan(c.z_mm) else f"{c.z_mm:4.2f}"
            lines.append(f"  #{i}  x={c.x_mm:+6.1f}  y={c.y_mm:+6.1f}  z={z}mm  fz={c.fz_n:6.2f} N")
    lines.append(f"  contacts: {len(contacts)}    (Ctrl+C 종료)")
    return lines


class LiveDisplay:
    """ANSI 제자리 갱신 — 커서 홈 + 라인별 clear-to-EOL + 아래 잔여 clear."""

    def __init__(self) -> None:
        print("\033[2J", end="")   # 시작 시 1회 전체 클리어

    def render(self, lines: list[str]) -> None:
        body = "\n".join(ln + "\033[K" for ln in lines)
        print("\033[H" + body + "\033[J", end="", flush=True)


class FrameLatch:
    """총 fz 최대 프레임을 유지(최적/최대 프레임 = B1)."""

    def __init__(self) -> None:
        self.best_total: float = -1.0
        self.frame_idx: int | None = None
        self.pred_map: np.ndarray | None = None
        self.contacts: list[Contact] | None = None

    def update(self, frame_idx: int, pred_map: np.ndarray, contacts: list[Contact]) -> float:
        total = float(sum(c.fz_n for c in contacts))
        if total > self.best_total:
            self.best_total = total
            self.frame_idx = frame_idx
            self.pred_map = pred_map.copy()
            self.contacts = list(contacts)
        return total

    def reset(self) -> None:
        self.best_total = -1.0
        self.frame_idx = None
        self.pred_map = None
        self.contacts = None
