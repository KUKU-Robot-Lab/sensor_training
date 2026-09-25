#!/usr/bin/env python3
"""통합 데모 대시보드 — 3센서(각각 contacts·theta·bending) 1창 동시 시연.

레이아웃(1 figure, GridSpec 2x3):
  [ S1 FLAT SATS ] [ S2 theta(deg) + restore map ] [ S3 DEFORMED SATS ]
  [ ------ spec footer (D10) ------- ] [ 16-taxel raw inset   ]

설계:
  - SATS 엔진 1개 공유(predict 상태무관) → GPU 1회 로드로 3센서 처리.
  - 센서별 SerialReader. 포트 'none'이면 해당 패널 OFFLINE(현재 1센서만 연결돼도 동작).
  - 포트 고정 권장: /dev/serial/by-id/*  (ttyACM 번호 뒤바뀜 방지).
  - 무접촉/접촉 판정은 demo_contacts.state_banner 로 3패널 통일.

키:
  b = bending 채널 재장착(밴딩 무접촉 상태서 theta+bent-baseline 캡처, ~1s)
  z = theta 채널 flat 재영점
  q = 종료

★센서 선택: --sensor v7 을 주면 그 센서의 SATS run(g025>all4>g01)을 쓴다.
  미지정 시 v6 기본값이라, v7 센서를 꽂고도 v6 SATS 로 추론하게 되니 반드시 지정할 것.

예(4090, 1센서를 bending 패널로 테스트):
  .venv/bin/python -m sats.inference.run_dashboard --contacts-port none --bending-port auto
예(3센서 전시):
  .venv/bin/python -m sats.inference.run_dashboard \
    --contacts-port /dev/serial/by-id/A --theta-port /dev/serial/by-id/B \
    --bending-port /dev/serial/by-id/C
"""
from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np

from sats.inference.contact_filter import ContactFilter, FilterConfig
from sats.inference.demo_contacts import (
    ContactState, StateBanner, extract_contacts, state_banner,
)
from sats.inference.inference_engine import SATSInferenceEngine
from sats.inference.run_demo import auto_detect_port
from sats.inference.z_calibration import ZCalibration

_ROOT = Path(__file__).resolve().parents[2]
_ROLES = ("contacts", "theta", "bending")


# 모델 탐색은 폴더 규약(runs/{sats,deform,theta}/<v*>/)으로 통일 — model_registry 참조.
from sats.inference.model_registry import (
    default_estimator, deform_restorers, estimators, list_sensors, resolve_sats_run,
    sats_runs,
)


class EngineCache:
    """센서별 SATS 엔진을 만들어 재사용한다.

    ★패널마다 다른 센서를 쓰려면 엔진도 패널별이어야 한다. 하나를 공유하면 v7 센서를
    v6 SATS 로 추론하게 되고, 캘리브레이션이 달라 조용히 틀린 맵이 나온다.
    같은 센서를 두 패널이 쓰는 경우가 흔하므로 캐시해 중복 로드를 막는다.
    """

    def __init__(self, args) -> None:
        self.args = args
        self._cache: dict[str, object] = {}

    def get(self, sensor: str | None, run_dir: str | None = None):
        key = run_dir or sensor or "_default"
        if key not in self._cache:
            from sats.inference.inference_engine import SATSInferenceEngine
            run = resolve_sats_run(sensor, run_dir)
            self._cache[key] = SATSInferenceEngine(
                run, device=self.args.device, indenter_diameter_mm=self.args.diameter)
            print(f"[dashboard] SATS 로드: {Path(run).name}")
        return self._cache[key]

    def get_run(self, ckpt_path: str):
        """체크포인트 경로로 엔진 로드(UI 의 '추론 파일' 선택 — registry 가 준 경로)."""
        return self.get(None, str(ckpt_path))


class SensorChannel:
    """한 센서(=한 역할)의 리더 + 상태 + 매 틱 표시 payload 산출.

    role: contacts | theta | bending. reader None 이면 OFFLINE.
    """

    def __init__(self, role: str, reader, engine, bi, z_calib, args, restorer=None,
                 sensor: str | None = None) -> None:
        self.role = role
        self.reader = reader
        self.engine = engine
        self.sensor = sensor
        self.bi = bi
        self.z_calib = z_calib
        self.args = args
        self.connected = reader is not None
        self.last_seq = 0
        # theta 상태
        self.theta0 = 0.0
        self.hist: collections.deque = collections.deque(maxlen=max(20, args.theta_smooth))
        # 변형 복원기(각도-프리) — bending: SATS 입력 보정 / theta: 복원맵 표시(S2)용.
        self._restorer_all = restorer if role in ("bending", "theta") else None
        self.restorer = self._restorer_all
        if self.restorer is not None and sensor is not None \
                and getattr(self.restorer, "sensor", None) not in (None, sensor):
            self.restorer = None                  # 다른 센서 전용 복원기는 붙이지 않는다
        self.restore_on = self.restorer is not None
        self.armed = False
        self.theta_fixed = 0.0
        self.bent_ref = np.zeros(16, np.float32)
        # ★무접촉 자동 재영점: 접촉 후 히스테리시스 잔류·열/기압 드리프트가 쌓이면
        #   fz 가 fz_on(0.30N)을 넘어 유령 접촉이 뜬다(무접촉 노이즈 자체는 0.003N 이지만
        #   눌렀다 뗀 뒤 잔류는 훨씬 크다 — 실기 관측). 접촉이 없다고 판정된 상태가
        #   1초 이상 지속되면 창 평균을 느린 EMA 로 흡수해 0 기준을 되돌린다.
        #   한계: fz_on 미만의 아주 약한 지속 접촉도 서서히 흡수된다(자동 영점의 고전적
        #   트레이드오프). 접촉이 잡히는 동안에는 절대 갱신하지 않는다.
        self.drift_ref = np.zeros(16, np.float32)
        self._quiet_t0: float | None = None
        self._drift_t: float | None = None       # 시간 기반 EMA 용 직전 갱신 시각
        self._last_win_mean: np.ndarray | None = None
        # ★진단 로그(--diag-log) — 잔상(유령) 원인 판정용 per-tick NDJSON.
        #   raw/drift/필터 전후 접촉/흡수기 판정 근거를 그대로 남긴다.
        self._absorb_info: dict = {}
        self._dropout_seen = False               # 이번 에피소드에 채널 포화(-60%↓) 발생
        self._frame_hist: collections.deque = collections.deque(maxlen=200)  # (t, raw16)
        self._quar = np.zeros(16, bool)          # 채널 격리(드롭아웃 회복 대기)
        self._quar_t0 = np.zeros(16)             # 격리 시작 시각
        self._quar_ok_t = np.zeros(16)           # 회복 관찰 시작 시각(0=미관찰)
        self._diag = None
        if getattr(args, "diag_log", None) and role in ("contacts", "bending"):
            self._diag = open(args.diag_log, "a", buffering=1)
        self._boost_until = 0.0                  # 접촉 해제 직후 빠른 흡수 구간
        self._ghost_hist: collections.deque = collections.deque(maxlen=260)
        # 접촉 필터(히스테리시스·디바운스·무접촉 리셋·스무딩) — heatmap 역할만
        self.cfilter = ContactFilter(FilterConfig(
            fz_on=args.fz_on, fz_off=args.fz_off, on_frames=args.on_frames,
            off_frames=args.off_frames, pos_smooth=args.pos_smooth)) \
            if role in ("contacts", "bending") else None
        self.busy: str = ""                       # "" 또는 진행 중 작업 라벨(UI 표시)

    # ── 런타임 연결/해제 (UI 버튼) ────────────────────────────────────────────
    def connect(self, port: str) -> str | None:
        """포트에 리더 연결(기존 연결은 해제). 성공 None, 실패 사유 문자열."""
        self.disconnect()
        try:
            if self.engine is None and self.role in ("contacts", "bending"):
                return "먼저 v* 와 run 을 선택하세요"
            wsize = self.engine.window_size if self.engine is not None else 10
            reader = _build_reader(port, self.args, wsize)
        except Exception as e:                     # 포트 열기 실패 등 — UI에 사유 표시
            return f"연결 실패: {e}"
        if reader is None:
            return "포트 탐지 실패"
        self.reader, self.connected, self.last_seq = reader, True, 0
        if getattr(self, "manual_dead", None) and hasattr(reader, "set_dead_channels"):
            reader.set_dead_channels(self.manual_dead)     # 재연결에도 유지
        return None

    def disconnect(self) -> None:
        """리더 중지·상태 초기화(필터·재장착·영점 포함)."""
        if self.reader is not None:
            try:
                self.reader.stop()
            except Exception:
                pass
        self.reader, self.connected = None, False
        self.armed, self.theta_fixed = False, 0.0
        self.theta0 = 0.0
        self.hist.clear()
        self.drift_ref = np.zeros(16, np.float32)
        self._quiet_t0 = None
        self._drift_t = None
        self._frame_hist.clear()
        self._dropout_seen = False
        if self.cfilter is not None:
            self.cfilter.reset()

    def set_dead_channels(self, spec: str) -> str | None:
        """UI 의 dead 입력('7,11,15' 1-based) 적용 — 리더 표시 통계에서 0 고정."""
        try:
            idx = sorted({int(x) - 1 for x in spec.replace(" ", "").split(",") if x})
        except ValueError:
            return f"형식 오류: {spec!r} (예: 7,11,15)"
        if any(i < 0 or i > 15 for i in idx):
            return "채널은 1~16 범위"
        self.manual_dead = tuple(idx)
        if self.reader is not None and hasattr(self.reader, "set_dead_channels"):
            self.reader.set_dead_channels(idx)
        return None

    def apply_theta_sensor(self, sensor: str, engines: "EngineCache") -> str | None:
        """S2: 센서 지정 + 그 센서의 SATS 엔진 로드(복원 결과를 SATS 맵으로 보이기 위함)."""
        self.sensor = sensor
        try:
            self.engine = engines.get(sensor)
        except SystemExit as e:
            return str(e).splitlines()[0] + " — 복원 SATS 맵 없이 진행"
        return None

    def apply_estimator(self, ckpt: str) -> str | None:
        """S2: theta estimator 교체(패널별 — 다른 센서의 각도 추정). ckpt=파일 경로."""
        from sats.bending.config import BendingConfig
        from sats.inference.bending_infer import BendingInference
        try:
            self.bi = BendingInference(ckpt, device=self.args.device,
                                       restorer=None, cfg=BendingConfig())
        except Exception as e:
            return f"estimator 로드 실패: {e}"
        self.theta0 = 0.0
        self.hist.clear()
        return None

    def apply_restorer(self, ckpt: str) -> str | None:
        """deform 복원기 교체(UI 선택, ckpt=파일 경로). 센서 불일치는 막지 않고 경고만 —
        사용자가 명시적으로 골랐다면 그 판단을 따른다."""
        from sats.inference.deform_restore import try_load
        r, msg = try_load(ckpt, device=self.args.device)
        if r is None:
            return msg
        self._restorer_all = self.restorer = r
        self.restore_on = True
        print(f"[dashboard] {self.role}: {msg}")
        if self.sensor and getattr(r, "sensor", None) not in (None, self.sensor):
            return f"주의 — 복원기는 {r.sensor} 학습본(현재 패널 {self.sensor})"
        return None

    def apply_run(self, sensor: str, ckpt_path: str, engines: "EngineCache") -> str | None:
        """이 패널의 (센서, 추론 체크포인트) 확정 — port→v*→추론파일 흐름 마지막 단계.

        ★복원기는 학습에 쓴 센서에서만 유효하다(채널 마스킹·캘리브레이션이 다름).
        다른 센서면 붙이지 않고 알린다 — 조용히 잘못된 보정을 하는 것보다 낫다.
        """
        try:
            self.engine = engines.get_run(ckpt_path)
        except Exception as e:
            return f"run 로드 실패: {e}"
        self.sensor = sensor
        if self.role == "bending" and self._restorer_all is not None:
            match = getattr(self._restorer_all, "sensor", None)
            self.restorer = self._restorer_all if match in (None, sensor) else None
            self.restore_on = self.restorer is not None
            if self.restorer is None:
                return f"복원기는 {match} 전용 — {sensor} 에는 적용하지 않음"
        return None

    def toggle_restore(self) -> bool:
        """변형 복원 ON/OFF. 데모의 '변형 → 정지 → 복원' 구분 동작에 쓴다."""
        if self.restorer is None:
            return False
        self.restore_on = not self.restore_on
        if self.cfilter is not None:
            self.cfilter.reset()                  # 전환 시 접촉 트랙 초기화
        return self.restore_on

    def reset(self) -> None:
        """역할별 리셋: contacts=필터, theta=재영점, bending=재장착(무접촉 유지)."""
        if self.role == "contacts":
            if self.cfilter is not None:
                self.cfilter.reset()
            if self._last_win_mean is not None:      # 즉시 재영점(잔류를 한 번에 흡수)
                self.drift_ref = self._last_win_mean.copy()
                print(f"[dashboard] {self.role} 재영점 — 잔류 "
                      f"{float(np.abs(self._last_win_mean).max()):.2f}% 흡수")
        elif self.role == "theta":
            self.busy = "re-zero..."
            try:
                self.rezero_theta()
            finally:
                self.busy = ""
        else:
            if self.restorer is not None:        # 복원기가 있으면 재장착 자체가 불필요
                self.restore_on = True
                if self.cfilter is not None:
                    self.cfilter.reset()
                return
            self.busy = "arming..."
            try:
                self.arm_bending()
            finally:
                self.busy = ""

    # ── 데이터 취득 ──────────────────────────────────────────────────────────
    def _latest(self):
        if self.reader is None:
            return None
        if hasattr(self.reader, "get_latest_window_with_seq"):
            win, seq = self.reader.get_latest_window_with_seq()
        else:
            win, seq = self.reader.get_latest_window(), self.last_seq + 1
        if win is None or seq == self.last_seq:
            return None
        self.last_seq = seq
        return win

    @property
    def baseline(self):
        return None if self.reader is None else self.reader.baseline

    # ── 재영점 / 재장착(키 이벤트) ────────────────────────────────────────────
    def rezero_theta(self) -> None:
        """flat 구간 theta 중앙값을 0으로(theta 채널)."""
        if self.bi is None or self.baseline is None:
            return
        vals = []
        t_end = time.time() + 0.6
        while time.time() < t_end:
            win = self._latest()
            if win is not None:
                vals.append(self.bi.theta_from_pct(win, demo_baseline=self.baseline))
            time.sleep(0.02)
        if vals:
            self.theta0 = float(np.median(vals))
            self.hist.clear()

    def arm_bending(self) -> None:
        """밴딩 무접촉 상태서 theta 고정 + bent-baseline(밴딩 패턴) 캡처(~1s)."""
        if self.bi is None or self.baseline is None:
            return
        base = np.asarray(self.baseline, float)
        thetas, frames = [], []
        for _ in range(30):
            win = self._latest()
            if win is not None:
                thetas.append(self.bi.theta_from_pct(win, demo_baseline=base))
                frames.append(np.asarray(win, np.float32).mean(0))
            time.sleep(0.03)
        self.theta_fixed = float(np.median(thetas)) if thetas else 0.0
        self.bent_ref = (np.median(frames, axis=0).astype(np.float32)
                         if frames else np.zeros(16, np.float32))
        if self.cfilter is not None:
            self.cfilter.reset()                     # 재장착 시 트랙·상태 초기화
        self.armed = True

    # ── 매 틱 payload ─────────────────────────────────────────────────────────
    def _idle_payload(self, banner: StateBanner) -> dict:
        return {"kind": "heatmap" if self.role != "theta" else "theta",
                "banner": banner, "pred_map": None, "contacts": [],
                "theta": None, "noise": None, "units": None}

    def poll(self) -> dict:
        """현재 표시 payload. kind ∈ {heatmap, theta}."""
        if not self.connected:
            return self._idle_payload(state_banner(None, connected=False))
        if self.busy:                              # 재영점/재장착 진행 중
            return self._idle_payload(StateBanner(ContactState.OFFLINE, self.busy.upper(), "#c8a200"))
        if not getattr(self.reader, "baseline_ready", True):   # flat baseline 수집 중
            prog = float(getattr(self.reader, "baseline_progress", 0.0)) * 100
            return self._idle_payload(StateBanner(ContactState.OFFLINE,
                                                  f"BASELINE {prog:.0f}% — hands off", "#c8a200"))
        if self.engine is None and self.role in ("contacts", "bending"):
            return self._idle_payload(StateBanner(ContactState.OFFLINE,
                                                  "SELECT SENSOR + RUN", "#c8a200"))
        win = self._latest()
        if self.role == "contacts":
            return self._poll_heatmap(win, theta_deg=None, bent=False)
        if self.role == "bending":
            theta = None if self.restorer is not None else (
                self.theta_fixed if self.armed else None)
            return self._poll_heatmap(win, theta_deg=theta, bent=self.armed)
        return self._poll_theta(win)

    _QUIET_SEC = 1.0          # 이 시간 무접촉이 지속돼야 재영점 시작
    # ★시간 기반 시정수 — 틱당 α 방식은 틱 레이트에 종속된다(구현 버그였음):
    #   느린 PC(i7, CPU 추론)에서 틱이 20→2Hz 로 떨어지면 같은 잔류 흡수에 10배
    #   (25초+)가 걸려 "무접촉인데 히스테리시스가 남는" 증상이 됨. dt 로 환산해
    #   어떤 머신에서도 같은 초 단위 거동을 보장한다.
    _DRIFT_TAU_S = 2.5        # 느린 흡수 시정수(구 α=0.02@20Hz 와 동일 거동)
    _BOOST_TAU_S = 0.33       # 접촉 해제 직후 2초(구 α=0.15@20Hz 와 동일 거동)
    # ★관찰 5초인 이유: 3초 창에서는 힘이 출렁이는 실접촉(사람 손)의 하강 국면이
    #   잔류 감쇠와 구분되지 않는다(실측 오탐). 5초 + "창 내 최대 fz 비증가" 조합이면
    #   출렁임은 다시 올라간 흔적이 남아 걸러지고, 잔류는 단조 감쇠라 통과한다.
    _GHOST_SPAN_S = 5.0       # 빠른 경로: 감쇠 관찰 시간
    _GHOST_DECAY = 0.7        # 창 시작 대비 이만큼(30%+) 감쇠해야 잔류
    _GHOST_POS_MM = 1.5       # 이 이상 움직이면 유령 아님
    _GHOST_FZ_MAX = 2.0       # 실측 잔류 최대 ~1N + 여유
    _GHOST_LONG_S = 10.0      # 느린 경로: 감쇠가 미미한 고착 잔류
    # ★비증가 허용치 — 저력 유령은 fz 0.34↔0.39N(±15%) 수준으로 출렁인다(실로그).
    #   고정 5% 는 이 노이즈에 계속 걸려 흡수를 영원히 막았다 → 절대 0.08N 바닥.
    _GHOST_FZ_TOL_N = 0.08
    # ★리바운드 경로 — 강압(>3.5N)으로 채널이 포화/드롭아웃(-60%↓)된 에피소드는
    #   해제 후 그 채널이 +4~5% 오버슈트로 복귀해 고정점 유령을 만든다(실로그 20s+).
    #   드롭아웃을 목격했다면 유령 확률이 압도적 → 3s 정지만으로 즉시 흡수.
    #   트레이드오프: 강압 직후 가볍게 3s 이상 정지 유지하는 실접촉은 오흡수될 수
    #   있다(다시 누르면 즉시 재검출).
    _DROPOUT_PCT = -60.0
    _REBOUND_SPAN_S = 3.0
    # ★프레임 동결 경로(범용 최후 방어) — 3센서 실험 로그에서 45.6s(fz 2.79N)·18.7s
    #   잔상이 생존: 다중 표시(n!=1)·fz>2N·대잔차(25~59%) 로 기존 경로 전제가 전부
    #   깨졌다. 공통 물리는 하나 — **유령은 raw 가 얼어 있다**(실접촉은 힘 떨림으로
    #   미세하게라도 움직임). raw 16채널이 노이즈 밴드 내 6s 정지 + 접촉 표시 지속
    #   → 접촉 수·fz 무관 흡수. 트레이드오프: 6s+ 초정밀 정지 실접촉은 오흡수
    #   (재누름 즉시 재검출) — "무접촉 잔상 금지" 우선의 의도적 편향.
    _FREEZE_SPAN_S = 6.0
    _FREEZE_PCT = 0.6         # 6s 창에서 채널별 (max-min) 이 전부 이 미만이면 동결
    # ★채널 격리(원천 차단) — 드롭아웃(-60%↓)을 겪은 채널은 회복 완료까지 수십 초간
    #   +5~85% 잔차를 뿜으며 유령을 반복 생성한다(3센서 실로그: 동결 흡수 후에도
    #   몇 초짜리 유령 재발). 그 채널은 물리적으로 신뢰 불가 → SATS 입력에서 잔차를
    #   0 으로 마스킹하고, raw 가 기준선 ±1.5% 로 2s 안정 복귀하면 해제.
    #   30s 넘게 안 돌아오면 새 기준선으로 수용(영구 마스킹 방지).
    _QUAR_OK_PCT = 1.5
    _QUAR_OK_SEC = 2.0
    _QUAR_MAX_S = 30.0
    # ★부호 게이트(즉시·최우선) — 이 센서는 누르면 raw 가 **음(-)** 으로 간다
    #   (드롭아웃 -100% 방향). 실로그 46개 에피소드 전수: 실접촉 33개는 전 틱에서
    #   min(resid) ≤ -8.8%, 유령들은 틱의 69~100% 에서 min > -3%(눌림 증거 없음,
    #   +5~25% 오버슈트만 존재 — 학습 분포 밖이라 SATS 가 접촉으로 오인).
    #   → 눌림 증거(-3% 이하 채널) 없는 접촉 표시는 틱 단위로 즉시 억제.
    _PRESS_MIN_PCT = -3.0

    def _update_drift_ref(self, has_contacts: bool, raw_mean: np.ndarray) -> None:
        now = time.perf_counter()
        dt = min(now - self._drift_t, 1.0) if self._drift_t is not None else 0.05
        self._drift_t = now
        if has_contacts:
            self._quiet_t0 = None
            return
        if self._quiet_t0 is None:
            self._quiet_t0 = now
        elif now - self._quiet_t0 > self._QUIET_SEC:
            self._dropout_seen = False           # 깨끗한 무접촉 복귀 → 에피소드 종료
            import math
            tau = self._BOOST_TAU_S if now < self._boost_until else self._DRIFT_TAU_S
            a = 1.0 - math.exp(-dt / tau)          # 틱 레이트 무관, 초 단위 시정수
            self.drift_ref = ((1.0 - a) * self.drift_ref
                              + a * raw_mean).astype(np.float32)

    def _maybe_absorb_ghost(self, contacts, raw_mean: np.ndarray) -> list:
        """★교착 해소 — 히스테리시스 잔류가 만든 유령 접촉을 감지해 즉시 흡수한다.

        구멍: 잔류 fz 가 fz_off(0.15N)를 넘는 동안 필터는 '접촉 중'이고, 접촉 중에는
        재영점이 갱신되지 않는다 → 잔류가 유령을 만들고 유령이 흡수를 막는 순환.
        유령의 물리적 시그니처로 끊는다: **제자리에 멈춘 채(1.5mm 미만) fz 가
        단조 감쇠**(점탄성 회복). 실접촉은 유지 시 fz 가 감쇠하지 않고, 움직이면
        위치가 변하므로 걸리지 않는다. 판정되면 잔류를 통째로 재영점하고 필터를 리셋.
        """
        now = time.perf_counter()
        # 프레임 동결 경로 — 접촉 수·fz 와 무관하게 raw 정지를 본다(최후 방어)
        self._frame_hist.append((now, raw_mean.copy()))
        if contacts and any(now - f[0] >= self._FREEZE_SPAN_S for f in self._frame_hist):
            arr = np.stack([f[1] for f in self._frame_hist
                            if now - f[0] <= self._FREEZE_SPAN_S])
            spread = float((arr.max(axis=0) - arr.min(axis=0)).max())
            if spread < self._FREEZE_PCT:
                self.drift_ref = raw_mean.copy()
                if self.cfilter is not None:
                    self.cfilter.reset()
                self._ghost_hist.clear()
                self._frame_hist.clear()
                self._dropout_seen = False
                self._absorb_info = {"why": "absorb_freeze", "spread": round(spread, 2),
                                     "n": len(contacts)}
                print(f"[dashboard] {self.role} 유령 흡수 — raw 동결 "
                      f"{self._FREEZE_SPAN_S:.0f}s(변화 {spread:.2f}%<{self._FREEZE_PCT}%)"
                      f"·표시 {len(contacts)}점")
                return []
        if len(contacts) != 1 or contacts[0].fz_n > self._GHOST_FZ_MAX:
            self._ghost_hist.clear()
            self._absorb_info = {"why": "n!=1" if len(contacts) != 1 else "fz>cap"}
            return contacts
        c = contacts[0]
        self._ghost_hist.append((now, c.x_mm, c.y_mm, c.fz_n))
        old = [h for h in self._ghost_hist if now - h[0] >= self._GHOST_SPAN_S]
        if not old:
            self._absorb_info = {"why": "span<5s"}
            return contacts
        t0, x0, y0, fz0 = old[-1]
        moved = max(abs(c.x_mm - x0), abs(c.y_mm - y0))
        fz_max_w = max(h[3] for h in self._ghost_hist)
        tol = max(self._GHOST_FZ_TOL_N, 0.05 * fz0)   # 저력 유령의 노이즈 흡수(절대 바닥)
        # 빠른 경로: 5초에 30%+ 감쇠(창 내 최대도 허용치 내 비증가 — 출렁임 배제)
        fast = (moved < self._GHOST_POS_MM and c.fz_n < self._GHOST_DECAY * fz0
                and fz_max_w <= fz0 + tol)
        # 느린 경로: 감쇠가 미미한 고착 잔류 — 10초 이상 정지 + fz 비증가.
        #   트레이드오프(명시): 사람이 10초 이상 완전히 정지한 채 누르고 있으면
        #   잔류로 오판될 수 있다. 데모에서 드문 패턴이고, 다시 누르면 즉시 재검출.
        oldest = self._ghost_hist[0]
        static_all = max(abs(c.x_mm - oldest[1]), abs(c.y_mm - oldest[2])) < self._GHOST_POS_MM
        slow = (now - oldest[0] >= self._GHOST_LONG_S and static_all
                and fz_max_w <= oldest[3] + max(self._GHOST_FZ_TOL_N, 0.05 * oldest[3]))
        # 리바운드 경로: 이번 에피소드에서 드롭아웃(-60%↓)을 봤다면 3s 정지로 즉시 흡수
        old3 = [h for h in self._ghost_hist if now - h[0] >= self._REBOUND_SPAN_S]
        rebound = (self._dropout_seen and bool(old3) and c.fz_n <= 1.2
                   and max(abs(c.x_mm - old3[-1][1]), abs(c.y_mm - old3[-1][2]))
                   < self._GHOST_POS_MM)
        # 진단 로그용 — 어떤 조건이 흡수를 막았는지 그대로 기록(추정 아님)
        self._absorb_info = {
            "why": ("absorb_rebound" if rebound else "absorb_fast" if fast
                    else "absorb_slow" if slow else "blocked"),
            "moved": round(moved, 2), "fz0": round(fz0, 3), "fz": round(c.fz_n, 3),
            "fzmax_ratio": round(fz_max_w / max(fz0, 1e-6), 3),
            "span": round(now - oldest[0], 1), "dropout": self._dropout_seen}
        if fast or slow or rebound:
            self._dropout_seen = False
            self.drift_ref = raw_mean.copy()         # 잔류 전체를 0 기준으로
            if self.cfilter is not None:
                self.cfilter.reset()
            self._ghost_hist.clear()
            why = ("드롭아웃 리바운드(강압 후)" if rebound
                   else f"fz {fz0:.2f}→{c.fz_n:.2f}N 감쇠" if fast
                   else f"{self._GHOST_LONG_S:.0f}s 정지·비증가(고착 잔류)")
            print(f"[dashboard] {self.role} 유령 접촉 흡수 — 정지 {moved:.1f}mm·{why}")
            return []
        return contacts

    def _update_quarantine(self, raw_mean: np.ndarray) -> np.ndarray:
        """드롭아웃 채널 격리 갱신 — 격리 마스크(bool[16]) 반환."""
        now = time.perf_counter()
        resid = raw_mean - self.drift_ref
        newly = (resid <= self._DROPOUT_PCT) & ~self._quar
        if newly.any():
            self._quar |= newly
            self._quar_t0[newly] = now
            self._quar_ok_t[newly] = 0.0
            chs = ",".join(f"S{i + 1:02d}" for i in np.flatnonzero(newly))
            print(f"[dashboard] {self.role} 채널 격리(드롭아웃): {chs} — 회복까지 마스킹")
        if self._quar.any():
            q = self._quar
            ok = q & (np.abs(resid) < self._QUAR_OK_PCT)
            self._quar_ok_t[q & ~ok] = 0.0                      # 회복 관찰 리셋
            start = q & ok & (self._quar_ok_t == 0.0)
            self._quar_ok_t[start] = now
            done = q & ok & (self._quar_ok_t > 0.0) \
                & (now - self._quar_ok_t >= self._QUAR_OK_SEC)
            # 장기 미복귀 → 새 기준선 수용(영구 마스킹 방지). ★눌리는 중(-3% 이하)에는
            #   수용 금지 — 눌린 값을 기준선으로 삼으면 이후 위치·힘이 통째로 틀어진다.
            stale = q & (now - self._quar_t0 >= self._QUAR_MAX_S) \
                & (resid > self._PRESS_MIN_PCT)
            if stale.any():
                self.drift_ref = self.drift_ref.copy()
                self.drift_ref[stale] = raw_mean[stale]
                done = done | stale
            if done.any():
                self._quar &= ~done
                chs = ",".join(f"S{i + 1:02d}" for i in np.flatnonzero(done))
                print(f"[dashboard] {self.role} 채널 격리 해제: {chs}")
        return self._quar.copy()

    def _diag_write(self, raw_mean: np.ndarray, pre, post) -> None:
        """per-tick 진단 한 줄(NDJSON) — sats/tools/ghost_log_analysis.py 로 분석."""
        import json as _json
        try:
            self._diag.write(_json.dumps({
                "t": round(time.time(), 3), "role": self.role,
                "raw": [round(float(v), 3) for v in raw_mean],
                "drift": [round(float(v), 3) for v in self.drift_ref],
                "pre": [[round(c.x_mm, 2), round(c.y_mm, 2), round(c.fz_n, 3)] for c in pre],
                "post": [[round(c.x_mm, 2), round(c.y_mm, 2), round(c.fz_n, 3)] for c in post],
                "ghost": self._absorb_info}) + "\n")
        except Exception:
            pass                                    # 진단이 데모를 죽여선 안 된다

    def _poll_heatmap(self, win, *, theta_deg, bent: bool) -> dict:
        a = self.args
        if win is None:
            banner = state_banner([], theta_deg=theta_deg, theta_band_deg=a.theta_deadband)
            return {"kind": "heatmap", "banner": banner, "pred_map": None,
                    "contacts": [], "theta": theta_deg, "units": None}
        raw_mean = np.asarray(win, np.float32).mean(0)
        self._last_win_mean = raw_mean
        # 강압 포화/드롭아웃 감지(-60%↓) — 해제 후 리바운드 유령 예고 신호(실로그 근거)
        if float((raw_mean - self.drift_ref).min()) <= self._DROPOUT_PCT:
            self._dropout_seen = True
        quar_mask = self._update_quarantine(raw_mean)
        win = (np.asarray(win, np.float32) - self.drift_ref[None, :])   # 드리프트 보정
        # ★격리 채널이라도 지금 눌리고 있으면(-3% 이하) 실신호를 통과시킨다.
        #   강한 실접촉 자체가 -60% 를 넘어 격리를 트리거하므로, 무조건 마스킹하면
        #   눌린 채널이 빠진 패턴으로 위치를 추정해 출력 위치가 크게 끌려간다(실기
        #   회귀 보고). 마스킹 대상은 해제 후 양(+)의 리바운드 꼬리뿐이다.
        mask_now = quar_mask & ((raw_mean - self.drift_ref) > self._PRESS_MIN_PCT)
        if mask_now.any():
            win[:, mask_now] = 0.0               # 회복 꼬리 잔차만 원천 차단
        # ★복원기 우선: 창마다 변형 오프셋을 추정해 뺀다(고정 스냅숏 방식보다 적응적).
        restored = None
        if self.role == "bending" and self.restorer is not None:
            restored = self.restorer.restore(np.asarray(win, np.float32))
        if restored is not None and self.restore_on:
            frame = restored
        else:
            frame = (win - self.bent_ref[None, :]).astype(np.float32) if bent else win
        pmap = self.engine.predict(frame)
        raw = extract_contacts(
            pmap, grid_min_mm=self.engine.grid_min_mm, grid_step_mm=self.engine.grid_step_mm,
            taxel_area=self.engine.taxel_area, diameter_mm=a.diameter,
            max_contacts=a.contacts, min_distance_mm=a.min_distance_mm,
            rel_threshold=a.rel_threshold, min_fz=a.min_fz,
            position_gain=getattr(a, "position_gain", 1.0), z_calib=self.z_calib)
        # 히스테리시스·디바운스·무접촉 리셋·스무딩(released 시 빈 리스트 → blank)
        if self.cfilter is not None:
            contacts, released = self.cfilter.update(raw)
            if released and not contacts:            # 해제 직후 = 잔류 최대 → 빠른 흡수
                self._boost_until = time.perf_counter() + 2.0
        else:
            contacts = raw
        contacts = self._maybe_absorb_ghost(contacts, raw_mean)
        # 부호 게이트: 눌림 증거 없는 표시는 유령 — 억제하고 드리프트 흡수에 맡긴다
        if contacts and float((raw_mean - self.drift_ref).min()) > self._PRESS_MIN_PCT:
            if self.cfilter is not None:
                self.cfilter.reset()
            self._absorb_info = {"why": "sign_gate",
                                 "resid_min": round(float((raw_mean - self.drift_ref).min()), 2)}
            if time.perf_counter() - getattr(self, "_sign_gate_last", 0.0) > 3.0:
                self._sign_gate_last = time.perf_counter()
                print(f"[dashboard] {self.role} 유령 억제(부호 게이트) — 눌림 증거 없음"
                      f"(min resid {self._absorb_info['resid_min']}% > {self._PRESS_MIN_PCT}%)")
            contacts = []
        self._update_drift_ref(bool(contacts), raw_mean)
        if self._diag is not None:
            self._diag_write(raw_mean, raw, contacts)
        banner = state_banner(contacts, theta_deg=theta_deg, theta_band_deg=a.theta_deadband)
        return {"kind": "heatmap", "banner": banner, "pred_map": pmap,
                "contacts": contacts, "theta": theta_deg,
                "units": win if self.role == "bending" else None,
                # ★복원 결과는 토글과 무관하게 항상 계산해 보여준다 — OFF 상태에서도
                #   "켜면 이렇게 된다"가 보여야 before/after 데모가 성립한다.
                "units_restored": restored,
                "note": ("RESTORE ON" if self.restore_on else "RESTORE OFF")
                        if self.restorer is not None else ""}

    def _poll_theta(self, win) -> dict:
        a = self.args
        if win is None or self.bi is None or self.baseline is None:
            banner = state_banner([], connected=self.connected)
            return {"kind": "theta", "banner": banner, "theta": None, "noise": None,
                    "units": None, "units_restored": None}
        theta_abs = self.bi.theta_from_pct(win, demo_baseline=self.baseline)
        val = theta_abs - self.theta0
        # ★유효밴드(20°) 미만은 노이즈 — 이력을 비워 미디언 지연 없이 즉시 0 으로 수렴.
        #   유지하면 큰 각도에서 내려올 때 이력의 잔상이 수 초간 남는다(동기화 어긋난 느낌).
        if abs(val) < a.theta_deadband:
            self.hist.clear()
        self.hist.append(val)
        sm = max(1, int(a.theta_smooth))
        recent = list(self.hist)[-sm:]
        smoothed = float(np.median(recent))
        noise = float(np.std(recent))
        display = 0.0 if abs(smoothed) < a.theta_deadband else smoothed   # 관측 하한 dead-band
        banner = state_banner([], theta_deg=display, theta_band_deg=a.theta_deadband)
        # ★S2 복원맵: 복원된 입력을 SATS 에 통과시킨 압력맵 — run_demo 2d 와 같은 형태.
        #   (4×4 taxel 값이 아니라 "복원 후 SATS 가 보는 세계"를 보여준다)
        restored = (self.restorer.restore(np.asarray(win, np.float32))
                    if self.restorer is not None else None)
        restored_map = (self.engine.predict(restored.astype(np.float32))
                        if restored is not None and self.engine is not None else None)
        return {"kind": "theta", "banner": banner, "theta": display, "noise": noise,
                "units": win, "units_restored": restored, "restored_map": restored_map}


def _build_reader(port: str, args, window_size: int):
    """port=='none' → None. 'auto' → 자동탐지. 그 외 → 실포트. (reader or None)."""
    if port == "none":
        return None
    if args.mock:
        from sats.inference.mock_reader import MockSensorReader
        r = MockSensorReader(window_size=window_size)
        r.start()
        return r
    resolved = auto_detect_port(args.baudrate) if port == "auto" else port
    if resolved is None:
        print(f"[dashboard] 포트 자동탐지 실패({port}) → 해당 패널 OFFLINE")
        return None
    from sats.inference.serial_reader import SensorSerialReader
    r = SensorSerialReader(port=resolved, baudrate=args.baudrate, window_size=window_size,
                           baseline_seconds=args.baseline_seconds, startup_delay=args.startup_delay,
                           protocol=args.protocol)
    r.start()
    print(f"[dashboard] {resolved}@{args.baudrate} 시작")
    return r


def _wait_baselines(channels, timeout: float = 30.0) -> None:
    """실센서 리더들의 baseline 준비 대기(손 떼고)."""
    readers = [c.reader for c in channels if c.reader is not None
               and hasattr(c.reader, "baseline_ready")]
    if not readers:
        return
    print("[dashboard] baseline 대기 — 모든 센서 손 떼고 평평 유지...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all(getattr(r, "baseline_ready", True) for r in readers):
            print("[dashboard] baseline 완료\n")
            return
        for r in readers:
            if getattr(r, "error_message", None):
                print(f"[dashboard] 리더 오류: {r.error_message}")
        time.sleep(0.3)
    print("[dashboard] baseline 타임아웃 — 진행(일부 패널 부정확 가능)\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="통합 데모 대시보드(3센서 1창)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ui", choices=["tk", "mpl"], default="tk",
                   help="tk=컨트롤 UI(포트·연결·리셋 버튼) / mpl=구 matplotlib 뷰")
    p.add_argument("--no-autoconnect", action="store_true",
                   help="[tk] 시작 시 자동 연결 안 함(UI에서 수동 연결)")
    p.add_argument("--fullscreen", action="store_true",
                   help="[tk] 전체화면으로 시작(전시용 1920x1080). F11 토글·Esc 해제")
    p.add_argument("--diag-log", default=None,
                   help="잔상(유령) 진단 NDJSON 경로 — per-tick raw/drift/접촉/흡수기 "
                        "판정 기록. 분석: python -m sats.tools.ghost_log_analysis <파일>")
    p.add_argument("--contacts-port", default="auto", help="contacts 센서 포트('none'=비활성)")
    p.add_argument("--theta-port", default="none", help="theta 센서 포트")
    p.add_argument("--bending-port", default="none", help="bending 센서 포트")
    p.add_argument("--sensor", default=None,
                   help="★센서 버전(예: v7) — SATS run 을 그 센서 것으로 자동 지정"
                        "(g025 > all4 > g01 순). 미지정 시 v6 기본값")
    p.add_argument("--run-dir", default=None,
                   help="SATS run 직접 지정(--sensor 보다 우선)")
    p.add_argument("--diameter", type=float, default=10.0, help="D10 기준(고정)")
    p.add_argument("--contacts", type=int, default=1,
                   help="최대 접촉 수(기본 1, 창에서 [c]로 1→2→3 토글)")
    p.add_argument("--z-calib", default=None)
    p.add_argument("--deform-restorer",
                   default=str(_ROOT / "sats/bending/runs/deform_restorer/best.pt"),
                   help="각도-프리 변형 복원기 체크포인트. 있으면 S3 가 재장착 없이 "
                        "매 창 baseline 을 복원한다([r] 로 ON/OFF 토글)")
    p.add_argument("--min-distance-mm", type=float, default=None)
    p.add_argument("--rel-threshold", type=float, default=0.3)
    p.add_argument("--min-fz", type=float, default=0.1)
    p.add_argument("--position-gain", type=float, default=1.0,
                   help="과대응답 보정(중심 기준 1/gain 축소). 실측 D10 gain=1.80, "
                        "기본 1.0=off(전면 검증 전)")
    # 접촉 필터 — 무접촉 노이즈 플로어(D10 fz p99≈0.003N=사실상 0)에 맞춘 기본값
    p.add_argument("--fz-on", type=float, default=0.20, help="접촉 ON 총 fz 임계(N)")
    p.add_argument("--fz-off", type=float, default=0.10, help="접촉 OFF 임계(N, 히스테리시스)")
    p.add_argument("--on-frames", type=int, default=2, help="ON 확정 연속 프레임")
    p.add_argument("--off-frames", type=int, default=5, help="OFF 확정 연속=무접촉 리셋")
    p.add_argument("--pos-smooth", type=int, default=3, help="위치 median 스무딩 프레임(1=off)")
    p.add_argument("--theta-smooth", type=int, default=7)
    p.add_argument("--theta-deadband", type=float, default=20.0)
    # 화면 방향(마운팅 보정) — 손 이동과 마커 이동 방향 일치
    p.add_argument("--flip-y", action=argparse.BooleanOptionalAction, default=True,
                   help="센서 +y가 물리적으로 아래쪽이면 화면도 아래로(기본 on, --no-flip-y로 해제)")
    p.add_argument("--flip-x", action=argparse.BooleanOptionalAction, default=False,
                   help="좌우 반전(기본 off)")
    p.add_argument("--viz-fps", type=float, default=10.0)
    p.add_argument("--infer-max-fps", type=float, default=20.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--mock", action="store_true", help="하드웨어 없이(연결된 포트 자리에 mock)")
    p.add_argument("--baudrate", type=int, default=250000)
    p.add_argument("--baseline-seconds", type=float, default=5.0)
    p.add_argument("--startup-delay", type=float, default=2.0)
    p.add_argument("--protocol", choices=["auto", "binary", "csv"], default="binary")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.min_distance_mm is None:
        args.min_distance_mm = args.diameter        # 단일접촉 peak-split 방지
    if args.device == "auto":                    # ★engine 이전에 해석 — tk 모드는 engine 이 없다
        import torch as _torch
        args.device = "cuda" if _torch.cuda.is_available() else "cpu"
    ports = {"contacts": args.contacts_port, "theta": args.theta_port, "bending": args.bending_port}

    engines = EngineCache(args)
    engine = None
    if args.ui != "tk":                      # mpl 뷰는 기존대로 즉시 로드
        print("[1/3] SATS 엔진 로드")
        engine = engines.get(args.sensor, args.run_dir)
        args.run_dir = str(resolve_sats_run(args.sensor, args.run_dir))
    z_path = args.z_calib or (Path(__file__).resolve().parent / "z_calibration_v6.json")
    z_calib = ZCalibration.load(z_path) if Path(z_path).exists() else None

    # tk UI는 theta/bending 패널을 언제든 연결할 수 있으므로 estimator 를 미리 로드하되,
    # ★가중치가 없어도 기동은 막지 않는다(fresh clone) — S2 의 est 콤보에서 나중에 선택.
    bi = None
    need_bi = args.ui == "tk" or ports["theta"] != "none" or ports["bending"] != "none"
    if need_bi:
        est = default_estimator()
        if est is None:
            print("[2/3] bending estimator 없음 — S2 est 콤보에서 선택하면 활성화됩니다"
                  f" (배치 위치: {_ROOT / 'runs/theta/<v*>/'})")
        else:
            try:
                from sats.bending.config import BendingConfig
                from sats.inference.bending_infer import BendingInference
                bi = BendingInference(est, device=args.device, restorer=None,
                                      cfg=BendingConfig())
                print(f"[2/3] bending estimator 로드: {est}")
            except Exception as e:
                print(f"[2/3] bending estimator 로드 실패({e}) — S2 에서 다시 선택 가능")

    # ★변형 복원기(각도-프리) — 있으면 S3 패널이 재장착 없이 매 창 baseline 을 되돌린다.
    #   없어도 기동은 계속된다(UI restore 콤보에서 선택).
    from sats.inference.deform_restore import try_load as _load_restorer
    restorer, msg = _load_restorer(args.deform_restorer, device=args.device)
    if restorer is None:
        for path in deform_restorers().values():     # 폴더 규약 + 구 위치 전부 순회
            restorer, msg = _load_restorer(path, device=args.device)
            if restorer is not None:
                break
    print(f"[2/3] {msg}")

    if args.ui == "tk":
        # UI에서 패널별 연결 — CLI 포트 지정이 있으면 초기 연결로 반영
        channels = [SensorChannel(role, None, engine, bi, z_calib, args, restorer, args.sensor)
                    for role in _ROLES]
        # ★UI-only 시작: 연결·엔진 로드는 UI 에서 port → v* → 추론 run 순으로 고른다.
        print("[dashboard] UI 시작 — 각 패널에서 port → sensor → run 선택 후 연결하세요")
        from sats.inference.dashboard_ui import DashboardApp
        app = DashboardApp(channels, engine, args, engines)
        try:
            app.run()
        finally:
            for c in channels:
                c.disconnect()
        return

    # ── 구 matplotlib 뷰(--ui mpl) ──
    if all(v == "none" for v in ports.values()):
        ports["contacts"] = "auto"                  # 기본: 1센서 → contacts 패널
        print("[dashboard] 포트 미지정 → contacts=auto 로 진행")
    print("[3/3] 리더 시작")
    channels = [SensorChannel(role, _build_reader(ports[role], args, engine.window_size),
                              engine, bi, z_calib, args, restorer, args.sensor)
                for role in _ROLES]
    _wait_baselines(channels)
    from sats.inference.dashboard import Dashboard
    dash = Dashboard(channels, engine, args)
    try:
        dash.run()
    finally:
        for c in channels:
            if c.reader is not None:
                c.reader.stop()


if __name__ == "__main__":
    main()
