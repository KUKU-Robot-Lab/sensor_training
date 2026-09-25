"""변형(deformation) 취득 전용 로거 — 각도-프리 baseline 복원 학습용.

`final_logger_gui.py` 기반이나 **DUE 16채널만** 기록한다(모터·loadcell·AFD 불필요).
각도 라벨을 쓰지 않으므로 지그·스테이지 없이 손으로 센서를 변형시키며 취득한다.

프로토콜(Enter 로 진행 · 상태머신):
    [1] BASE_HEAD  무접촉·무하중 평평 (~10초)   → **빈 Enter** 로 다음 단계
    [2] DEFORM     손으로 순수 변형 (시간 표시)  → **빈 Enter** 로 다음 단계
                   (숫자 1~8 / 텍스트 + Enter = 구간 라벨 마커, 단계는 계속)
    [3] BASE_TAIL  다시 무접촉·무하중 (10초)     → **자동 종료·저장**
    앞뒤 baseline 두 점으로 **선형 드리프트 보정**이 가능해진다. 실제 단계 전환 시각은
    session_meta.json(stage_times_s)에 기록되어 분석 로더가 그대로 사용한다.

저장 경로(분석 스크립트가 바로 읽는 규약):
    skin_ws/raw_data/deform/<version>/<sNN_YYYYmmdd_HHMMSS>/due_v2_*.bin + session_meta.json
    → 학습: `python -m sats.bending.train_deform_restorer --deform-root skin_ws/raw_data/deform/v1`
    (`deform_data.discover_sessions` 가 하위 폴더를 재귀 탐색하므로 세션을 계속 쌓기만 하면 된다)

경로는 **이 파일 위치 기준 상대경로**로 해석하므로 어느 작업 디렉터리에서 실행해도 같다.

실행:
    python deform_logger_gui.py --version v1                 # 실센서
    python deform_logger_gui.py --version v1 --mock          # 하드웨어 없이 UI 점검
    python deform_logger_gui.py --version v1 --deform-sec 300 --port COM11
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

if sys.platform == "win32":
    os.system("")                                   # ANSI 활성화

# ── Qt 바인딩(플로팅을 안 쓰므로 pyqtgraph 불필요) ──────────────────────────
QT_BINDING = None
for _mod in ("PyQt5", "PyQt6", "PySide6", "PySide2"):
    try:
        QtCore = __import__(f"{_mod}.QtCore", fromlist=["QtCore"])
        QtWidgets = __import__(f"{_mod}.QtWidgets", fromlist=["QtWidgets"])
        QT_BINDING = _mod
        break
    except ImportError:
        continue
HAS_GUI = QT_BINDING is not None
if not HAS_GUI:
    print("Error: Qt 바인딩 없음 — pip install PyQt5 (또는 PyQt6/PySide6)")


def _qt_enum(name: str, scope: str):
    """평면 열거형(PyQt5/PySide2)과 중첩 열거형(PyQt6/PySide6) 모두 지원."""
    return getattr(QtCore.Qt, name) if hasattr(QtCore.Qt, name) \
        else getattr(getattr(QtCore.Qt, scope), name)


# ── 하드웨어 상수(final_logger_gui.py 와 동일 규약) ──────────────────────────
DEFAULT_PORT = "auto"                               # ★자동 탐지(실패 시 직접 지정)
DUE_BAUD_RATE = 250000
NUM_SENSORS = 16
FIFO_FRAMES = 10
PAYLOAD_SIZE = NUM_SENSORS * FIFO_FRAMES * 4        # 640 bytes

# ── 프로토콜 기본값(초) ─────────────────────────────────────────────────────
BASE_SEC = 10.0          # BASE_TAIL 자동 종료 길이 · BASE_HEAD 권장 길이
MIN_BASE_SEC = 5.0       # BASE_HEAD 최소 확보(너무 빨리 Enter 치는 것 방지)
_NO_DATA_SEC = 3.0       # 이만큼 프레임이 없으면 경고(포트는 열렸는데 데이터가 없는 경우)

_STAGES = ("BASE_HEAD", "DEFORM", "BASE_TAIL", "DONE")
_STAGE_MSG = {
    "BASE_HEAD": "① 손 떼고 평평하게 — 무접촉·무하중  [Enter=다음]",
    "DEFORM":    "② 손으로 변형! 다양하게 (정지도 섞어서)  [Enter=종료]",
    "BASE_TAIL": "③ 다시 손 떼고 평평하게 — 10초 후 자동 종료",
    "DONE":      "완료 — 저장됨",
}
_STAGE_COLOR = {"BASE_HEAD": "#c8a200", "DEFORM": "#c22", "BASE_TAIL": "#c8a200", "DONE": "#0a7a4f"}
# 변형 단계에서 순환 표시할 가이드(다양성 확보 — 학습 분포 커버리지가 성패를 가름)
_TIPS = [
    "위로 휘기 → 아래로 휘기",
    "좌우로 휘기",
    "★비틀기(torsion) — 각도 방식이 못 하던 변형",
    "한쪽 끝만 휘기",
    "변형한 채로 3~5초 정지 유지",
    "천천히 연속 변형",
    "빠르게 변형",
    "★극단 변형 — 데모에서 쓸 최대보다 조금 더 강하게",
]
# ★변형 크기 커버리지 구간 — 학습 평가(train_deform_restorer._MAG_BINS)와 동일 경계.
# 실측(v7 3세션): 0~10% 에 표본 81% 가 몰려 10~25% 는 17%, 25~50% 는 1.8% 뿐이었고,
# 그 결과 25% 초과에서 억제율이 음수까지 떨어졌다. 취득 중에 이 분포를 보고
# **부족한 구간을 채우는 것**이 다음 세션의 목적이다.
# ★sats/bending/deform_data.py 의 MAG_BINS 와 동일하게 유지할 것(분석과 같은 눈금).
# 손으로 파손 직전까지 변형해도 센서 전체 평균 |pct| 는 10% 를 넘지 않는다 — 그 이상은
# 파손 영역이다. 변형은 접촉과 달리 넓게 퍼지므로 **채널 평균**이 실제 휨을 나타낸다.
_MAG_BINS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, float("inf")))
# 세션당 **목표 시간(초)**. 비율은 세션이 길어질수록 희석되어 "언제 끊을지"의 기준이
# 될 수 없다. 학습에 필요한 것은 각 구간의 절대 윈도우 수이므로 초로 관리한다.
_MAG_TARGET_SEC = (45.0, 60.0, 60.0, 0.0)   # 강한 쪽(5-10%)을 더 두껍게
_FS = 200.0

# 터미널 숫자 입력 → 구간 라벨 프리셋(위 가이드와 1:1)
_SEGMENT_PRESETS = {
    "1": "bend_up_down", "2": "bend_left_right", "3": "twist", "4": "bend_one_end",
    "5": "hold_static", "6": "slow_continuous", "7": "fast", "8": "extreme",
}


def parse_dead_channels(spec: str) -> list[int]:
    """'7,11,15'(1-based) → 0-based 인덱스. 빈 문자열이면 빈 목록.

    ★자동 탐지는 쓰지 않는다. 파손 taxel 은 raw=0 → 시작 시 baseline 이 0 으로 잡히고,
    이후 값이 정상 범위로 돌아오면 pct 가 수억 % 로 튀어(실측 6600606060%) 어떤 밴드
    기준으로도 안정적으로 걸러지지 않는다. 사람이 지정하는 편이 확실하다.
    """
    out = []
    for tok in spec.replace(" ", "").split(","):
        if not tok:
            continue
        if not tok.isdigit() or not 1 <= int(tok) <= NUM_SENSORS:
            raise SystemExit(f"--dead-channels 는 1~{NUM_SENSORS} 의 쉼표 목록이어야 함: {spec!r}")
        out.append(int(tok) - 1)
    return sorted(set(out))


def magnitude_coverage(hist: list[float]) -> list[float]:
    """|pct|max 이력 → 구간별 시간 비중[4]. 합이 1(빈 이력이면 전부 0)."""
    if not hist:
        return [0.0] * len(_MAG_BINS)
    counts = [sum(1 for m in hist if lo <= m < hi) for lo, hi in _MAG_BINS]
    return [c / len(hist) for c in counts]


def coverage_seconds(hist: list[float]) -> list[float]:
    """|pct|max 이력 → 구간별 **누적 초**. 200Hz 프레임 단위 집계 기준."""
    return [sum(1 for m in hist if lo <= m < hi) / _FS for lo, hi in _MAG_BINS]


def coverage_text(hist: list[float]) -> str:
    """구간별 누적 초(비율)를 한 줄로. 목표 미달 구간에 ★.

    ★이력이 비어 있을 때 0 을 나열하면 "변형 단계가 아직 아님"과 "그 구간을 안 했음"이
    구분되지 않는다. 그래서 비어 있으면 그 사실을 그대로 알린다.
    """
    if not hist:
        return "(변형 단계 ②부터 집계됩니다)"
    names = ("0-2", "2-5", "5-10", "10+")
    secs, cov = coverage_seconds(hist), magnitude_coverage(hist)
    done = all(sc >= t for sc, t in zip(secs, _MAG_TARGET_SEC))
    body = " · ".join(
        f"{'★' if sc < t else ''}{n}%: {sc:3.0f}s({c * 100:2.0f}%)"
        for n, sc, c, t in zip(names, secs, cov, _MAG_TARGET_SEC))
    return body + ("   ✔목표 달성 — 종료해도 좋습니다" if done else "")

is_running = True
due_queue: Queue = Queue()
flag_queue: Queue = Queue()          # 터미널 Enter → 구간 flag(ns)
reader_errors: Queue = Queue()
log_start_ns = 0


def elapsed_ns() -> int:
    return time.perf_counter_ns() - log_start_ns


def payload_to_rows(payload: bytes) -> list[list[int]]:
    v = struct.unpack("<" + ("I" * NUM_SENSORS * FIFO_FRAMES), payload)
    return [[v[s * FIFO_FRAMES + f] for s in range(NUM_SENSORS)] for f in range(FIFO_FRAMES)]


def _is_legacy_uart(device: str) -> bool:
    """/dev/ttyS0~31 = 메인보드 레거시 UART. Due 가 여기 붙는 일은 없다.

    ★후보에 남겨두면 33개까지 불어나고, 이들의 권한 오류가 진단을 오염시켜
    실제 원인(ACM 포트에서 프레임이 안 옴)을 가린다.
    """
    import re as _re
    return bool(_re.fullmatch(r"/dev/ttyS\d+", device))


def _port_score(p) -> int:
    """Arduino Due 로 보이는 포트에 높은 점수(정렬 우선순위)."""
    text = " ".join(str(x or "") for x in (p.description, p.manufacturer, p.product)).lower()
    score = 0
    if getattr(p, "vid", None) == 0x2341:            # Arduino VID
        score += 100
    if "arduino" in text or "due" in text:
        score += 50
    if "acm" in p.device.lower() or "usbmodem" in p.device.lower():
        score += 10
    if "bluetooth" in text or "/dev/ttys" in p.device.lower():
        score -= 100                                 # 가상·내장 포트는 뒤로
    return score


def _has_due_frame(data: bytes) -> bool:
    """0xAA + 640B + 0x55 프레임이 실제로 흐르는지 확인(오탐 방지)."""
    return any(data[i] == 0xAA and data[i + 1 + PAYLOAD_SIZE] == 0x55
               for i in range(len(data) - PAYLOAD_SIZE - 1))


class _RawPort:
    """list_ports 가 놓친 /dev 장치를 같은 인터페이스로 감싸는 최소 표현."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.description = "(glob)"
        self.manufacturer = self.product = None
        self.vid = None


def _candidate_ports() -> list:
    """list_ports + /dev glob 을 합쳐 중복 없이 점수순으로 정렬."""
    ports: list = []
    try:
        from serial.tools import list_ports
        ports = list(list_ports.comports())
    except ImportError:
        pass
    seen = {p.device for p in ports}
    if sys.platform != "win32":                      # list_ports 가 비어도 장치는 있을 수 있다
        import glob as _glob
        for dev in sorted(_glob.glob("/dev/ttyACM*") + _glob.glob("/dev/ttyUSB*")):
            if dev not in seen:
                ports.append(_RawPort(dev))
    return sorted(ports, key=_port_score, reverse=True)


def _diagnose(failures: list[tuple[str, Exception]]) -> None:
    """열기 실패 원인을 사용자가 바로 고칠 수 있는 조치로 번역."""
    joined = " ".join(str(e).lower() for _, e in failures)
    if "permission" in joined or "denied" in joined:
        print("  → ★권한 문제입니다. 다음 실행 후 **재로그인**하세요:")
        print("      sudo usermod -aG dialout $USER")
        print("    (임시 조치: sudo chmod a+rw /dev/ttyACM0)")
    if "busy" in joined or "resource temporarily unavailable" in joined:
        print("  → 다른 프로그램이 포트를 점유 중입니다(로거·시리얼 모니터 중복 실행?).")
        print("      확인: sudo fuser -v /dev/ttyACM0")


def autodetect_port(baud: int, probe_sec: float = 2.0) -> str | None:
    """후보 포트를 점수순으로 열어 **DUE 프레임이 오는 포트**를 고른다.

    포트 이름만 보고 고르면 다른 USB 시리얼 장치를 잡을 수 있으므로, 실제로 프레이밍이
    맞는 데이터가 흐르는지까지 확인한다. 실패 시 원인을 진단해 출력하고 None.
    """
    try:
        import serial
    except ImportError:
        print("Error: pyserial 없음 — pip install pyserial")
        return None
    cands = [p for p in _candidate_ports() if not _is_legacy_uart(p.device)]
    if not cands:
        print("[port] 시리얼 포트가 하나도 보이지 않습니다 — USB 연결·전원을 확인하세요.")
        print("      확인: ls -l /dev/ttyACM*   ·   dmesg | tail")
        return None
    failures: list[tuple[str, Exception]] = []
    for p in cands:
        print(f"[port] 탐색 {p.device} ({p.description}) ...", flush=True)
        try:
            with serial.Serial(p.device, baud, timeout=0.2) as ser:
                ser.reset_input_buffer()
                buf, t0 = b"", time.perf_counter()
                while time.perf_counter() - t0 < probe_sec:
                    buf += ser.read(4096)
                    if _has_due_frame(buf):
                        print(f"[port] ★DUE 발견 → {p.device}")
                        return p.device
                print(f"[port]   DUE 프레임 없음({len(buf)}B 수신)"
                      + ("  ← 데이터는 오는데 프레이밍 불일치(펌웨어/보드레이트 확인)"
                         if buf else "  ← 수신 자체가 없음"))
        except Exception as e:                       # 권한·점유 등은 다음 후보로
            print(f"[port]   열기 실패: {e}")
            failures.append((p.device, e))
    print(f"[port] 후보 {len(cands)}개 모두 실패.")
    if failures:
        _diagnose(failures)
    else:
        # 열리기는 했는데 DUE 프레임이 없었던 경우 — 권한이 아니라 장치 문제다
        print("  → 포트는 정상적으로 열렸으나 **DUE 프레임이 오지 않습니다**. 권한 문제가 아닙니다.")
        print("     1) Due 전원·USB 케이블(데이터용인지) 확인")
        print("     2) Due 리셋 버튼 · USB 재연결 후 5초 대기")
        print("     3) 다른 프로그램이 이전에 포트를 잡고 있었다면 종료: pgrep -af logger")
        print("     4) 펌웨어가 올라가 있는지(다른 스케치로 덮이지 않았는지) 확인")
    return None


def due_reader(port: str, baud: int) -> None:
    """DUE 시리얼 → due_queue. 0xAA + 640B + 0x55 프레이밍."""
    try:
        import serial
        ser = serial.Serial(port, baud, timeout=0.1)
        ser.reset_input_buffer()
        while is_running:
            header = ser.read(1)
            if not header or header[0] != 0xAA:
                continue
            payload = ser.read(PAYLOAD_SIZE)
            footer = ser.read(1)
            if len(payload) == PAYLOAD_SIZE and footer and footer[0] == 0x55:
                due_queue.put((elapsed_ns(), payload))
        ser.close()
    except Exception as e:
        reader_errors.put(("DUE", str(e)))


def stdin_flag_listener() -> None:
    """터미널 입력 → 구간(segment) 시작 마커.

    입력 규약(입력 시각부터 그 라벨의 구간이 시작):
      1~8 + Enter : 프리셋 라벨(_SEGMENT_PRESETS, 화면 가이드와 동일 번호)
      텍스트 + Enter : 자유 라벨(예: twist_hold)
      빈 Enter    : 라벨 없는 경계(구간만 나눔)
    """
    print("[입력] ★빈 Enter = 다음 단계 진행")
    print("[입력] 변형 중 구간 라벨: 1=상하휨 2=좌우휨 3=비틀림 4=한쪽끝 5=정지유지 "
          "6=느린연속 7=빠름 8=극단 · 텍스트=자유라벨")
    while is_running:
        try:
            line = sys.stdin.readline()
        except Exception:
            return
        if not line:                     # EOF(파이프 등)
            return
        key = line.strip()
        t_s = elapsed_ns() / 1e9
        if key == "":                                # ★빈 Enter = 다음 단계로
            flag_queue.put((t_s, None))
            continue
        label = _SEGMENT_PRESETS.get(key, key)       # 숫자→프리셋, 그 외=자유 라벨
        flag_queue.put((t_s, label))
        print(f"[segment] {t_s:7.1f}s → '{label}' 구간 시작")


def mock_reader() -> None:
    """하드웨어 없이 UI·저장 경로 점검용 합성 신호(200Hz)."""
    import random
    base = [6_900_000] * NUM_SENSORS
    t0 = time.perf_counter()
    while is_running:
        t = time.perf_counter() - t0
        amp = 0.02 * (1 if 10 < t < 250 else 0.0)
        rows = []
        for f in range(FIFO_FRAMES):
            k = amp * (1 + 0.5 * random.random())
            rows.append([int(base[s] * (1 + k * ((s % 4) - 1.5) / 3) + random.gauss(0, 300))
                         for s in range(NUM_SENSORS)])
        flat = [rows[f][s] for s in range(NUM_SENSORS) for f in range(FIFO_FRAMES)]
        due_queue.put((elapsed_ns(), struct.pack("<" + "I" * len(flat), *flat)))
        time.sleep(FIFO_FRAMES / 200.0)


class BinaryBuffer:
    """DUE_V2 버퍼 기록(final_logger_gui 와 동일 포맷: magic 줄 + [ns(u64) + payload])."""

    def __init__(self, binfile, payload_size: int, magic: str = "DUE_V2") -> None:
        self.binfile = binfile
        self.payload_size = payload_size
        self.offset = 0
        self.binfile.write(f"{magic}\n".encode("ascii"))
        self.buffer = bytearray((8 + payload_size) * 8192)

    def write(self, ns: int, payload: bytes) -> None:
        size = 8 + len(payload)
        if self.offset + size > len(self.buffer):
            self.flush()
        struct.pack_into("<Q", self.buffer, self.offset, ns)
        self.buffer[self.offset + 8: self.offset + 8 + len(payload)] = payload
        self.offset += size

    def flush(self) -> None:
        if self.offset:
            self.binfile.write(self.buffer[:self.offset])
            self.offset = 0


def repo_paths() -> tuple[Path, Path]:
    """이 파일 기준 상대경로로 (skin_ws, raw_data) 반환 — 실행 위치 무관."""
    skin_ws = Path(__file__).resolve().parents[3]          # gui/ → tools/ → acquisition_code/ → skin_ws/
    return skin_ws, skin_ws / "raw_data"


def next_session_dir(raw_data: Path, version: str) -> Path:
    """raw_data/deform/<version>/testN/ 생성 — 기존 취득 폴더 규약(test*)과 동일, 순번 자동."""
    root = raw_data / "deform" / version
    root.mkdir(parents=True, exist_ok=True)
    used = [int(p.name[4:]) for p in root.iterdir()
            if p.is_dir() and p.name.startswith("test") and p.name[4:].isdigit()]
    n = max(used, default=0) + 1
    d = root / f"test{n}"
    d.mkdir()
    return d


if HAS_GUI:
    class DeformLoggerGUI(QtWidgets.QMainWindow):
        """프로토콜 상태머신 + 실시간 16-taxel 표시 + 변형 다양성 피드백."""

        def __init__(self, session_dir: Path, args) -> None:
            super().__init__()
            self.session_dir = session_dir
            self.args = args
            self.stage = "IDLE"
            self.stage_t0 = 0.0
            self.baseline: list[float] | None = None
            self.count = 0
            self.segments: list[dict] = []           # [{t_s, label}] 터미널 입력 구간 마커
            self.stage_times: dict[str, float] = {}  # 단계 전환 시각(초) — 프로토콜 경계 명시
            self.deform_max = 0.0                    # 변형 중 도달한 |pct| 최대(다양성 지표)
            self.deform_hist: list[float] = []
            self.last_frame_t = time.perf_counter()  # 프레임 흐름 감시
            self._no_data_warned = False
            self.dead_ch: list[int] = list(getattr(args, "dead", []))   # 통계 제외(기록은 16채널)
            self.live_ch = [i for i in range(NUM_SENSORS) if i not in self.dead_ch]
            self.last_flush = time.perf_counter()
            self._init_ui()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.due_path = session_dir / f"due_v2_{ts}.bin"
            self.f_due = open(self.due_path, "wb")
            self.due_bin = BinaryBuffer(self.f_due, PAYLOAD_SIZE)
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.tick)
            self.timer.start(50)

        # ── UI ──────────────────────────────────────────────────────────────
        def _init_ui(self) -> None:
            self.setWindowTitle(f"Deformation Logger — {self.session_dir.name}")
            self.resize(980, 760)
            c = QtWidgets.QWidget(); self.setCentralWidget(c)
            v = QtWidgets.QVBoxLayout(c)

            self.stage_lbl = QtWidgets.QLabel("시작 버튼을 누르세요")
            self.stage_lbl.setStyleSheet("font-weight: bold; font-size: 21px;")
            self.stage_lbl.setAlignment(_qt_enum("AlignCenter", "AlignmentFlag"))
            v.addWidget(self.stage_lbl)

            self.tip_lbl = QtWidgets.QLabel("")
            self.tip_lbl.setStyleSheet("font-size: 16px; color: #0033cc;")
            self.tip_lbl.setAlignment(_qt_enum("AlignCenter", "AlignmentFlag"))
            v.addWidget(self.tip_lbl)

            self.bar = QtWidgets.QProgressBar(); self.bar.setTextVisible(True)
            v.addWidget(self.bar)

            grid = QtWidgets.QGridLayout()
            self.boxes = []
            for r in range(4):
                for cc in range(4):
                    i = r * 4 + cc
                    b = QtWidgets.QLabel(f"S{i+1:02d}\n0.0%")
                    b.setAlignment(_qt_enum("AlignCenter", "AlignmentFlag"))
                    b.setStyleSheet("border:2px solid gray; font-size:18px; background:#f0f0f0;")
                    b.setMinimumSize(110, 80)
                    grid.addWidget(b, r, cc)
                    self.boxes.append(b)
            v.addLayout(grid)

            self.seg_lbl = QtWidgets.QLabel("현재 구간: (미지정 — 터미널에서 1~8/텍스트 입력)")
            self.seg_lbl.setStyleSheet("font-size: 14px; color: #7a3ca3; font-weight: bold;")
            v.addWidget(self.seg_lbl)

            self.stat_lbl = QtWidgets.QLabel("frames 0 | |pct|max 0.0%")
            self.stat_lbl.setStyleSheet("font-family: monospace; font-size: 13px;")
            self.stat_lbl.setWordWrap(True)          # 두 줄이 잘리지 않게
            v.addWidget(self.stat_lbl)

            row = QtWidgets.QHBoxLayout()
            self.start_btn = QtWidgets.QPushButton("시작 (Space / Enter)")
            self.start_btn.setFixedHeight(46); self.start_btn.clicked.connect(self.start_or_advance)
            self.abort_btn = QtWidgets.QPushButton("중단·저장")
            self.abort_btn.setFixedHeight(46); self.abort_btn.clicked.connect(self.finish)
            row.addWidget(self.start_btn); row.addWidget(self.abort_btn)
            v.addLayout(row)

        def keyPressEvent(self, e) -> None:
            if e.key() in (_qt_enum("Key_Space", "Key"), _qt_enum("Key_Return", "Key"),
                           _qt_enum("Key_Enter", "Key")):
                self.start_or_advance()

        def start_or_advance(self) -> None:
            """창에서도 시작·다음단계 가능(터미널 빈 Enter 와 동일)."""
            if self.stage == "IDLE":
                self.start()
            else:
                self.advance()

        # ── 상태머신 ─────────────────────────────────────────────────────────
        def start(self) -> None:
            global log_start_ns
            if self.stage != "IDLE":
                return
            log_start_ns = time.perf_counter_ns()
            while True:                              # ★시작 전 큐에 쌓인 프레임 폐기
                try:                                 # (log_start_ns 이전 기준의 타임스탬프라
                    due_queue.get_nowait()           #  기록되면 세션 시간축이 깨진다)
                except Empty:
                    break
            self._enter("BASE_HEAD")
            self.start_btn.setText("다음 단계 (Enter)")

        def _enter(self, stage: str) -> None:
            self.stage = stage
            self.stage_t0 = time.perf_counter()
            self.stage_times[stage] = round(elapsed_ns() / 1e9, 3)   # 프로토콜 경계 기록
            self.stage_lbl.setText(_STAGE_MSG[stage])
            self.stage_lbl.setStyleSheet(
                f"font-weight:bold; font-size:21px; color:{_STAGE_COLOR[stage]};")
            if stage != "DEFORM":
                self.tip_lbl.setText("")
            if stage == "BASE_TAIL":
                self.start_btn.setEnabled(False)     # 이후 자동 종료
            if stage == "DONE":
                self.finish()

        def advance(self) -> None:
            """빈 Enter(또는 버튼) → 다음 단계. BASE_TAIL 은 자동 종료라 수동 전환 없음."""
            el = time.perf_counter() - self.stage_t0
            if self.stage == "BASE_HEAD":
                if el < MIN_BASE_SEC:
                    print(f"[안내] baseline 최소 {MIN_BASE_SEC:.0f}초 필요 "
                          f"(현재 {el:.1f}s) — 조금 더 기다렸다 Enter")
                    return
                self._enter("DEFORM")
            elif self.stage == "DEFORM":
                self._enter("BASE_TAIL")

        def _stage_len(self) -> float | None:
            """자동 종료 길이. BASE_TAIL 만 자동, 앞 두 단계는 Enter 대기(None)."""
            return self.args.base_sec if self.stage == "BASE_TAIL" else None

        def _check_data_flow(self) -> None:
            """프레임이 끊기면 알린다.

            ★포트가 열려도 프레임이 안 올 수 있다(Due 정지·프로그래밍 포트 연결·펌웨어
            미동작). 이때 reader 는 예외를 내지 않으므로 오류 표시도 없고, 화면은 baseline
            이 변하지 않은 채 조용히 멈춘 것처럼 보인다. 실제로 이 상태로 세션을 끝까지
            진행해 빈 데이터를 얻은 사례가 있어 감시가 필요하다.
            """
            if self.stage not in ("BASE_HEAD", "DEFORM", "BASE_TAIL"):
                return
            gap = time.perf_counter() - self.last_frame_t
            if gap < _NO_DATA_SEC:
                self._no_data_warned = False
                return
            if not self._no_data_warned:
                self._no_data_warned = True
                print(f"[경고] {gap:.0f}초째 DUE 프레임 없음 — 센서/포트를 확인하세요.", flush=True)
                print("  1) 다른 프로그램이 포트를 잡고 있는지: sudo fuser -v /dev/ttyACM0")
                print("  2) Due 를 뽑았다 꽂거나 리셋 버튼")
                print("  3) --port 를 빼고 실행하면 자동 탐지가 프레임 유무까지 확인합니다")
            self.stage_lbl.setText(f"⚠ DUE 프레임 없음 {gap:.0f}s — 터미널 안내 참조")
            self.stage_lbl.setStyleSheet("color:#c22; font-weight:bold; font-size:18px;")

        # ── 루프 ────────────────────────────────────────────────────────────
        def tick(self) -> None:
            if not reader_errors.empty():
                who, msg = reader_errors.get()
                print(f"[오류] {who}: {msg}", flush=True)       # ★터미널에도 — 창만 보면 놓친다
                self.stage_lbl.setText(f"{who} 오류: {msg}")
                self.stage_lbl.setStyleSheet("color:#c22; font-weight:bold; font-size:18px;")
                return
            self._check_data_flow()
            while not flag_queue.empty():            # 터미널 입력 처리
                t_s, label = flag_queue.get()
                if label is None:                    # 빈 Enter = 다음 단계
                    self.advance()
                elif self.stage in ("BASE_HEAD", "DEFORM", "BASE_TAIL"):
                    self.segments.append({"t_s": round(t_s, 3), "label": label})
                    self.seg_lbl.setText(f"현재 구간: {label}  (총 {len(self.segments)}개)")
            latest = None
            while True:                              # 큐 비우며 기록
                try:
                    ns, payload = due_queue.get_nowait()
                except Empty:
                    break
                self.last_frame_t = time.perf_counter()
                if self.stage in ("BASE_HEAD", "DEFORM", "BASE_TAIL"):
                    self.due_bin.write(ns, payload)
                    self.count += 1
                rows = payload_to_rows(payload)
                latest = rows[-1]
                if self.baseline is None and self.count > 20:
                    self.baseline = [sum(col) / len(col) for col in zip(*rows)]
                if self.stage == "DEFORM" and self.baseline:
                    # ★프레임 단위로 집계한다. 틱당 1개만 쌓으면 표본이 지나치게 성기고
                    # 짧게 스쳐간 강한 변형이 통계에서 사라진다.
                    for r in rows:                    # ★채널 평균 = 센서 전체가 얼마나 휘었는가
                        self.deform_hist.append(sum(
                            abs((r[i] - (self.baseline[i] or 1)) / (self.baseline[i] or 1) * 100.0)
                            for i in self.live_ch) / len(self.live_ch))

            if latest and self.baseline:
                pct = [((latest[i] - (self.baseline[i] or 1)) / (self.baseline[i] or 1)) * 100.0
                       for i in range(NUM_SENSORS)]
                self._paint(pct)
                mx = sum(abs(pct[i]) for i in self.live_ch) / len(self.live_ch)  # 채널 평균
                if self.stage == "DEFORM":
                    self.deform_max = max(self.deform_max, mx)
                self.stat_lbl.setText(
                    f"frames {self.count:7d} | 변형 {mx:5.2f}% | "
                    f"deform peak {self.deform_max:5.1f}%\n"
                    f"커버리지({self.stage})  {coverage_text(self.deform_hist)}   (★=부족)"
                    + ("\n⚠ 10% 초과 — 손가락으로 **누르고** 있습니다. 변형만 주세요"
                       if self.stage == "DEFORM" and mx > 10.0 else "")
                    + (f"\n★파손 채널 제외: "
                       f"{','.join(f'S{i + 1:02d}' for i in self.dead_ch)}"
                       if self.dead_ch else ""))

            if self.stage in ("BASE_HEAD", "DEFORM", "BASE_TAIL"):
                el = time.perf_counter() - self.stage_t0
                total = self._stage_len()
                if total is None:                    # Enter 대기 단계 — 경과시간만
                    self.bar.setMaximum(0)           # busy indicator
                    hint = ("Enter 로 다음" if el >= MIN_BASE_SEC or self.stage == "DEFORM"
                            else f"최소 {MIN_BASE_SEC:.0f}s")
                    self.bar.setFormat(f"{self.stage}  {el:5.1f}s   ({hint})")
                else:
                    self.bar.setMaximum(int(total)); self.bar.setValue(int(min(el, total)))
                    self.bar.setFormat(f"{self.stage}  {el:.0f}/{total:.0f}s  (자동 종료)")
                    if el >= total:
                        self._enter(_STAGES[_STAGES.index(self.stage) + 1])
                if self.stage == "DEFORM":
                    self.tip_lbl.setText("▶ " + _TIPS[int(el // 20) % len(_TIPS)])

            now = time.perf_counter()
            if now - self.last_flush >= 1.0:
                self.due_bin.flush(); self.f_due.flush(); self.last_flush = now

        def _paint(self, pct: list[float]) -> None:
            for i, p in enumerate(pct):
                self.boxes[i].setText(f"S{i+1:02d}\n{p:+.1f}%")
                v = min(255, int(abs(p) * 10))
                if p > 1.0:
                    s = f"background-color: rgb(255,{255-v},{255-v}); border:2px solid red;"
                elif p < -1.0:
                    s = f"background-color: rgb({255-v},{255-v},255); border:2px solid blue;"
                else:
                    s = "background-color:#f0f0f0; border:2px solid gray;"
                self.boxes[i].setStyleSheet(s + " font-size:18px;")

        # ── 종료·메타 ────────────────────────────────────────────────────────
        def finish(self) -> None:
            global is_running
            if self.stage == "FINISHED":
                return
            self.due_bin.flush(); self.f_due.flush()
            hist = self.deform_hist
            meta = {
                "version": self.args.version,
                "session": self.session_dir.name,
                "created": datetime.now().isoformat(timespec="seconds"),
                "protocol": {"base_sec": self.args.base_sec, "advance": "enter"},
                "frames": self.count,
                "deform_peak_pct": round(self.deform_max, 2),
                "deform_median_pct": round(float(sorted(hist)[len(hist) // 2]), 2) if hist else 0.0,
                "dead_channels_1based": [i + 1 for i in self.dead_ch],   # 통계 제외(기록은 16채널)
                "magnitude_coverage": {                 # 구간별 시간 비중(학습 평가와 동일 경계)
                    n: round(c, 4) for n, c in
                    zip(("0-2", "2-5", "5-10", "10+"), magnitude_coverage(hist))},
                "magnitude_seconds": {                  # 구간별 누적 초(취득량 판단 기준)
                    n: round(sc, 1) for n, sc in
                    zip(("0-2", "2-5", "5-10", "10+"), coverage_seconds(hist))},
                "stage_times_s": self.stage_times,   # 프로토콜 단계 전환 시각(초)
                "segments": self.segments,           # [{t_s,label}] 변형 구간 마커(터미널 입력)
                "flags_s": [g["t_s"] for g in self.segments],   # 하위호환(시각만)
                "stage_reached": self.stage,
                "due_file": self.due_path.name,
                "mock": bool(self.args.mock),
            }
            (self.session_dir / "session_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            self.stage = "FINISHED"
            print(f"[deform] 커버리지 {coverage_text(hist)}")
            over = coverage_seconds(hist)[3]
            if over > 3.0:
                print(f"  ⚠ 10% 초과 {over:.0f}s — 손가락 압박(접촉)이 섞였을 가능성. "
                      f"학습에서 자동 제외되지만, 활성면을 누르지 말고 끝단만 잡으세요.")
            short = [f"{n}%({t - sc:.0f}s 부족)" for n, sc, t in
                     zip(("0-2", "2-5", "5-10"), coverage_seconds(hist), _MAG_TARGET_SEC)
                     if sc < t]
            if short:
                print(f"  ★목표 미달: {' · '.join(short)} — 다음 세션에서 그 강도로 "
                      f"**더 오래 유지**하세요(순간적으로 찍는 것보다 유지가 표본을 만듭니다).")
            self.stage_lbl.setText(
                f"저장 완료 — frames {self.count}, deform peak {self.deform_max:.1f}%")
            self.stage_lbl.setStyleSheet("color:#0a7a4f; font-weight:bold; font-size:20px;")
            self.tip_lbl.setText(str(self.session_dir))
            is_running = False

        def closeEvent(self, event) -> None:
            self.finish()
            try:
                self.f_due.close()
            except Exception:
                pass
            event.accept()


def main() -> None:
    if not HAS_GUI:
        return
    p = argparse.ArgumentParser(description="변형 취득 전용 로거(DUE only)")
    p.add_argument("--version", default=None,
                   help="버전 폴더명(예: v7). 미지정 시 시작할 때 터미널에서 입력받음")
    p.add_argument("--port", default=DEFAULT_PORT,
                   help="'auto'(기본)=DUE 프레임이 흐르는 포트를 자동 탐지 · 또는 직접 지정")
    p.add_argument("--baud", type=int, default=DUE_BAUD_RATE)
    p.add_argument("--base-sec", type=float, default=BASE_SEC, help="뒤 baseline 자동 종료 길이(초)")
    p.add_argument("--dead-channels", default="",
                   help="파손 taxel(1-based, 예 '7,11,15'). 변형 크기 통계에서 제외한다. "
                        "기록되는 bin 은 항상 16채널 그대로다(마스킹은 학습 시점).")
    p.add_argument("--mock", action="store_true", help="하드웨어 없이 UI·경로 점검")
    args = p.parse_args()

    args.dead = parse_dead_channels(args.dead_channels)
    if args.dead:
        print(f"[deform] 통계 제외 채널: "
              f"{','.join(f'S{i + 1:02d}' for i in args.dead)} (기록은 16채널 그대로)")
    if not args.mock and args.port == "auto":        # ★포트 자동 탐지
        found = autodetect_port(args.baud)
        if not found:
            print("Error: DUE 포트를 찾지 못했습니다 — 연결·전원 확인 후 재시도하거나 "
                  "--port /dev/ttyACM0 처럼 직접 지정하세요.")
            return
        args.port = found

    if not args.version:                             # 시작 시 버전 입력(v* 분리)
        v = input("버전 입력 (예: v7): ").strip()
        args.version = v if v else "v1"

    _, raw_data = repo_paths()
    session_dir = next_session_dir(raw_data, args.version)
    print(f"[deform] 세션 폴더: {session_dir}")
    print(f"[deform] 포트: {'mock' if args.mock else args.port}")
    print(f"[deform] 프로토콜: ①baseline —Enter→ ②변형(자유) —Enter→ "
          f"③baseline {args.base_sec:.0f}s → 자동 종료")

    target = mock_reader if args.mock else (lambda: due_reader(args.port, args.baud))
    threading.Thread(target=target, daemon=True).start()
    threading.Thread(target=stdin_flag_listener, daemon=True).start()

    app = QtWidgets.QApplication(sys.argv)
    gui = DeformLoggerGUI(session_dir, args)
    gui.show()
    sys.exit(getattr(app, "exec", app.exec_)())


if __name__ == "__main__":
    main()
