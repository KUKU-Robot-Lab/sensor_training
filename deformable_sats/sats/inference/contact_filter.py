"""실시간 접촉 필터 — 히스테리시스·디바운스·무접촉 리셋·위치 스무딩.

프레임별 raw 접촉(extract_contacts 결과)을 상태머신으로 안정화한다:
  - 히스테리시스: 총 fz ≥ fz_on 이면 ON, < fz_off 면 OFF(깜빡임 방지).
  - 디바운스: on_frames/off_frames 연속 만족해야 상태 전환.
  - 무접촉 리셋: OFF 확정 시 빈 리스트 + released 신호(표시·latch 초기화용).
  - 위치 스무딩: CONTACT 동안 트랙별 최근 median (분해능 한계 내 떨림 제거).

임계값은 분해능/무접촉 노이즈 플로어에서 정한다(scripts/measure_resolution.py, 4090).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from sats.inference.demo_contacts import Contact


@dataclass(frozen=True)
class FilterConfig:
    """접촉 필터 파라미터(총 fz[N] 기준)."""
    fz_on: float = 0.30       # 접촉 ON 임계(무접촉 노이즈 플로어 + 마진)
    fz_off: float = 0.15      # 접촉 OFF 임계(히스테리시스, on의 ~절반)
    on_frames: int = 2        # ON 확정 연속 프레임(디바운스)
    off_frames: int = 5       # OFF 확정 연속 프레임(릴리즈=무접촉 리셋)
    pos_smooth: int = 3       # 트랙 위치/fz median 스무딩 프레임(1=off)
    match_mm: float = 4.0     # 프레임간 트랙 매칭 최대 거리


def _med(dq) -> float:
    return float(np.median(np.asarray(dq, float)))


def _nanmed(dq) -> float:
    a = np.asarray(dq, float)
    return float("nan") if np.all(np.isnan(a)) else float(np.nanmedian(a))


class ContactFilter:
    """총 fz 히스테리시스 상태머신 + 트랙 위치 스무딩.

    update(contacts) -> (filtered, released):
      filtered  : CONTACT 상태일 때 안정화된 접촉(그 외 빈 리스트)
      released  : 이번 프레임에 OFF 확정(무접촉 리셋 발생) 여부
    """

    def __init__(self, cfg: FilterConfig | None = None) -> None:
        self.cfg = cfg or FilterConfig()
        self.reset()

    def reset(self) -> None:
        self.contact = False
        self._on = 0
        self._off = 0
        self._tracks: list[dict] = []

    def update(self, contacts: list[Contact]) -> tuple[list[Contact], bool]:
        total = float(sum(c.fz_n for c in contacts)) if contacts else 0.0
        cfg = self.cfg
        if not self.contact:                       # NO_CONTACT → ON 감시
            self._on = self._on + 1 if total >= cfg.fz_on else 0
            if self._on >= cfg.on_frames:
                self.contact = True; self._off = 0
            else:
                return [], False
        else:                                      # CONTACT → OFF 감시(히스테리시스)
            self._off = self._off + 1 if total < cfg.fz_off else 0
            if self._off >= cfg.off_frames:
                self.reset()
                return [], True                    # 릴리즈 = 무접촉 리셋
        return self._smooth(contacts), False

    def _smooth(self, contacts: list[Contact]) -> list[Contact]:
        S = max(1, int(self.cfg.pos_smooth))
        if S <= 1 or not contacts:
            return contacts
        used = [False] * len(self._tracks)
        out, new_tracks = [], []
        for c in contacts:
            j = self._nearest(c, used)
            tr = self._tracks[j] if j >= 0 else {k: deque(maxlen=S)
                                                 for k in ("x", "y", "z", "fz", "pv")}
            if j >= 0:
                used[j] = True
            tr["x"].append(c.x_mm); tr["y"].append(c.y_mm); tr["z"].append(c.z_mm)
            tr["fz"].append(c.fz_n); tr["pv"].append(c.peak_val)
            new_tracks.append(tr)
            out.append(Contact(x_mm=_med(tr["x"]), y_mm=_med(tr["y"]), z_mm=_nanmed(tr["z"]),
                               fz_n=_med(tr["fz"]), peak_val=_med(tr["pv"])))
        self._tracks = new_tracks
        return out

    def _nearest(self, c: Contact, used: list[bool]) -> int:
        best_j, best_d = -1, self.cfg.match_mm
        for j, tr in enumerate(self._tracks):
            if used[j]:
                continue
            d = math.hypot(c.x_mm - tr["x"][-1], c.y_mm - tr["y"][-1])
            if d < best_d:
                best_d, best_j = d, j
        return best_j
