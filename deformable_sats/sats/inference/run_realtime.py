"""
sats/inference/run_realtime.py

실시간 SATS 추론 + 시각화 진입점.

실행 예시
---------
# 2D 시각화 (기본)
python3 -m sats.inference.run_realtime \
    --run-dir sats/training/runs/04.25-sats-test4-e2e_v1 \
    --port /dev/ttyACM0

# 3D 시각화
python3 -m sats.inference.run_realtime \
    --run-dir sats/training/runs/04.25-sats-test4-e2e_v1 \
    --port /dev/ttyACM0 \
    --mode 3d

# 센서 200Hz 최대 반영 (헤드리스 고속 추론)
python3 -m sats.inference.run_realtime \
    --run-dir sats/training/runs/04.25-sats-test4-e2e_v1 \
    --port /dev/ttyACM0 \
    --mode none \
    --protocol binary \
    --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Protocol, Tuple

import matplotlib.pyplot as plt
import numpy as np


class _ReaderWithSeq(Protocol):
    def get_latest_window_with_seq(self) -> tuple[Optional[np.ndarray], int]:
        ...


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SATS 실시간 추론 시각화",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        required=True,
        help="학습 run 디렉터리 (config.json + best_model.pt 포함)",
    )
    p.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="Arduino 시리얼 포트",
    )
    p.add_argument(
        "--baudrate",
        type=int,
        default=250_000,
        help="시리얼 통신 속도",
    )
    p.add_argument(
        "--mode",
        choices=["none", "2d", "3d", "both"],
        default="2d",
        help="시각화 모드 (none이면 헤드리스 추론)",
    )
    p.add_argument(
        "--query-xy",
        nargs=2,
        type=float,
        metavar=("X_MM", "Y_MM"),
        default=None,
        help="초기 query 위치 (mm). 예: --query-xy 0.0 0.0\n"
        "실행 중 맵 클릭으로도 변경 가능.",
    )
    p.add_argument(
        "--baseline-seconds",
        type=float,
        default=5.0,
        help="baseline 자동 측정 시간 (초). 이 기간 동안 무접촉 상태 유지.",
    )
    p.add_argument(
        "--startup-delay",
        type=float,
        default=3.0,
        help="포트 오픈 후 Arduino 리셋 대기 시간 (초). Due 는 포트 오픈 시 리셋됨.",
    )
    p.add_argument(
        "--protocol",
        choices=["auto", "binary", "csv"],
        default="binary",
        help="시리얼 입력 프로토콜. 최신 vensor.ino 는 binary 사용(권장).",
    )
    p.add_argument(
        "--indenter-diameter-mm",
        type=float,
        default=5.0,
        help="크기입력(A) 모델용 고정 인덴터 지름 조건(mm). 실시간에선 접촉 크기를 "
             "모르므로 d5 디폴트 — 위치는 크기 오지정에도 강건, magnitude 만 영향.",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="(Deprecated) 과거 스케일용 vmax 옵션. 새 옵션: --vmax-mode / --vmax-nmm2",
    )
    p.add_argument(
        "--vmax-mode",
        choices=["gt-max", "manual"],
        default="gt-max",
        help="시각화 color scale 방식. gt-max는 GT 전체 최대값 고정, manual은 --vmax-nmm2 사용.",
    )
    p.add_argument(
        "--vmax-nmm2",
        type=float,
        default=None,
        help="manual 모드에서 사용할 color scale 최대값 (N/mm²).",
    )
    p.add_argument(
        "--viz-threshold-nmm2",
        type=float,
        default=0.01,
        help="시각화 맵에서 이 값 미만(N/mm²)을 0으로 클리핑. <=0 이면 비활성.",
    )
    p.add_argument(
        "--viz-fps",
        type=float,
        default=20.0,
        help="시각화 갱신 속도 (Hz).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="(Deprecated) --viz-fps 사용 권장",
    )
    p.add_argument(
        "--infer-max-fps",
        type=float,
        default=0.0,
        help="추론 상한 Hz. <=0 이면 센서 신규 윈도우가 들어오는 즉시 추론.",
    )
    p.add_argument(
        "--report-interval-seconds",
        type=float,
        default=2.0,
        help="성능 리포트 출력 주기(초).",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="추론 장치 (auto | cuda | cpu)",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="실제 센서 없이 모의 데이터로 테스트 (시리얼 포트 불필요)",
    )
    return p


def _get_latest_window_with_seq(reader: object, last_seq: int) -> tuple[Optional[np.ndarray], int]:
    if hasattr(reader, "get_latest_window_with_seq"):
        seq_reader = reader  # type: ignore[assignment]
        return seq_reader.get_latest_window_with_seq()

    # 호환용 fallback: seq API가 없는 구현은 매 호출마다 새 seq로 간주
    basic_reader = reader  # type: ignore[assignment]
    win = basic_reader.get_latest_window()
    if win is None:
        return None, last_seq
    return win, last_seq + 1


def main() -> None:
    args = _build_parser().parse_args()

    if args.fps is not None:
        print("[경고] --fps 는 deprecated 입니다. --viz-fps 로 대체합니다.")
        args.viz_fps = args.fps

    run_dir = Path(args.run_dir)
    query_xy: Optional[Tuple[float, float]] = (
        tuple(args.query_xy) if args.query_xy else None  # type: ignore[assignment]
    )

    from sats.inference.serial_reader import SensorSerialReader
    from sats.inference.inference_engine import SATSInferenceEngine, TAXEL_AREA
    from sats.inference.viz_postprocess import (
        MODEL_OUTPUT_SCALE,
        apply_viz_threshold_nmm2,
        compute_gt_global_max_nmm2,
        to_nmm2,
    )

    print("\n[1/3] 추론 엔진 초기화...")
    engine = SATSInferenceEngine(
        run_dir=run_dir,
        device=args.device,
        indenter_diameter_mm=args.indenter_diameter_mm,
    )
    print(f"  config path: {run_dir / 'config.json'}")
    print(f"  model path : {run_dir / 'best_model.pt'}")

    vmax_nmm2: Optional[float] = None
    if args.vmax_nmm2 is not None:
        vmax_nmm2 = args.vmax_nmm2
        print(f"[scale] manual vmax 적용: {vmax_nmm2:.6f} N/mm²")
    elif args.vmax is not None:
        print("[경고] --vmax 는 deprecated 입니다. --vmax-nmm2 사용을 권장합니다.")
        if args.vmax > 2.0:
            vmax_nmm2 = args.vmax / MODEL_OUTPUT_SCALE
            print(
                f"[scale] legacy 스케일(vmax={args.vmax:.4f})로 판단하여 "
                f"{vmax_nmm2:.6f} N/mm² 로 변환합니다."
            )
        else:
            vmax_nmm2 = args.vmax
            print(f"[scale] --vmax 값을 N/mm² 로 해석: {vmax_nmm2:.6f}")
    elif args.vmax_mode == "manual":
        raise ValueError("manual 모드에서는 --vmax-nmm2 (또는 legacy --vmax) 를 지정해야 합니다.")
    else:
        gt_dir = Path(engine.cfg.gt_dir)
        if not gt_dir.exists():
            gt_dir = (_ROOT / gt_dir).resolve()
        vmax_nmm2 = compute_gt_global_max_nmm2(str(gt_dir))
        print(f"[scale] GT global max 고정: {vmax_nmm2:.6f} N/mm²")

    if args.mock:
        from sats.inference.mock_reader import MockSensorReader

        print("\n[2/3] Mock 리더 시작 (--mock 모드)")
        reader = MockSensorReader(window_size=engine.window_size)
        reader.start()
        print("  mock baseline 즉시 완료\n")
    else:
        print(f"\n[2/3] 시리얼 리더 시작: {args.port} @ {args.baudrate} baud")
        reader = SensorSerialReader(
            port=args.port,
            baudrate=args.baudrate,
            window_size=engine.window_size,
            baseline_seconds=args.baseline_seconds,
            startup_delay=args.startup_delay,
            protocol=args.protocol,
        )
        reader.start()

        print(
            f"\n  !! 센서에서 손 떼고 기다려 주세요 "
            f"(startup {args.startup_delay:.0f}s + baseline {args.baseline_seconds:.0f}s) !!\n"
        )
        t_start = time.time()
        while not reader.baseline_ready:
            if reader.error_message:
                print(f"\n[오류] 시리얼 리더 오류: {reader.error_message}")
                reader.stop()
                sys.exit(1)
            elapsed = time.time() - t_start
            print(
                f"\r  [{elapsed:5.1f}s]  bursts={reader.bursts_received:4d}"
                f"  bytes_in={reader.raw_bytes_in:6d}"
                f"  footer_err={reader.footer_errors:3d}"
                f"  parse_err={reader.parse_errors:3d}"
                f"  baseline {reader.baseline_progress*100:.0f}%   ",
                end="",
                flush=True,
            )
            time.sleep(0.5)
        print("\n  baseline 완료!\n")

    print("[3/3] 시각화 창 초기화...")

    viz2d = None
    viz3d = None

    if args.mode in ("2d", "both"):
        from sats.inference.realtime_2d import RealtimeViz2D

        viz2d = RealtimeViz2D(engine=engine, query_xy_mm=query_xy, vmax=vmax_nmm2)

    if args.mode in ("3d", "both"):
        from sats.inference.realtime_3d import RealtimeViz3D

        viz3d = RealtimeViz3D(engine=engine, query_xy_mm=query_xy, vmax=vmax_nmm2)

    viz_interval = 0.0 if args.viz_fps <= 0 else 1.0 / args.viz_fps
    infer_interval = 0.0 if args.infer_max_fps <= 0 else 1.0 / args.infer_max_fps

    print("\n실시간 추론 중... (창 닫기 또는 Ctrl+C 로 종료)\n")
    print(f"{'Infer':>8}  {'Peak x':>8}  {'Peak y':>8}  {'Peak P':>12}  {'Fz':>10}")
    print("-" * 58)

    last_window_seq = 0
    last_infer_time = 0.0
    last_viz_time = 0.0
    last_report_time = time.time()

    total_data_windows = 0
    total_infers = 0
    total_viz = 0

    report_data_windows = 0
    report_infers = 0
    report_viz = 0

    low_visibility_warned = False

    latest_pred_map_nmm2: Optional[np.ndarray] = None
    latest_viz_map_nmm2: Optional[np.ndarray] = None
    latest_viz_peak: Optional[Tuple[float, float, float]] = None
    latest_fz_viz: float = 0.0

    has_gui = args.mode != "none"

    try:
        while True:
            now = time.time()

            if has_gui:
                open_2d = viz2d.is_open() if viz2d else True
                open_3d = viz3d.is_open() if viz3d else True
                if not (open_2d or open_3d):
                    print("\n창이 닫혔습니다. 종료합니다.")
                    break
            else:
                open_2d = False
                open_3d = False

            if hasattr(reader, "error_message") and getattr(reader, "error_message"):
                print(f"\n[오류] 시리얼 리더 오류: {getattr(reader, 'error_message')}")
                break

            window, window_seq = _get_latest_window_with_seq(reader, last_window_seq)
            is_new_window = window is not None and window_seq != last_window_seq

            did_work = False

            if is_new_window:
                total_data_windows += window_seq - last_window_seq
                report_data_windows += window_seq - last_window_seq

                # 상한이 지정된 경우, 최신 윈도우까지만 당겨오고 추론은 rate-limit
                if infer_interval <= 0.0 or (now - last_infer_time) >= infer_interval:
                    pred_map_scaled = engine.predict(window)
                    peak_val_scaled = float(pred_map_scaled.max())
                    peak_val_nmm2 = peak_val_scaled / MODEL_OUTPUT_SCALE
                    pred_map_nmm2 = to_nmm2(pred_map_scaled)
                    viz_map_nmm2 = apply_viz_threshold_nmm2(
                        pred_map_nmm2,
                        args.viz_threshold_nmm2,
                    )

                    has_contact = float(viz_map_nmm2.max()) > 0.0
                    viz_peak = engine.get_peak(viz_map_nmm2) if has_contact else None
                    fz_viz = float(viz_map_nmm2.clip(0).sum()) * engine.taxel_area

                    if (
                        not low_visibility_warned
                        and args.viz_threshold_nmm2 > 0
                        and float(viz_map_nmm2.max()) <= 0.0
                        and peak_val_nmm2 > 1e-4
                    ):
                        print(
                            "[경고] 시각화 맵이 임계값으로 모두 제거되었습니다. "
                            f"--viz-threshold-nmm2 값을 더 낮춰 보세요 "
                            f"(현재={args.viz_threshold_nmm2:.4f}, peak={peak_val_nmm2:.5f})."
                        )
                        low_visibility_warned = True

                    latest_pred_map_nmm2 = pred_map_nmm2
                    latest_viz_map_nmm2 = viz_map_nmm2
                    latest_viz_peak = viz_peak
                    latest_fz_viz = fz_viz

                    total_infers += 1
                    report_infers += 1
                    last_infer_time = now
                    did_work = True

                    if total_infers % 10 == 0:
                        if viz_peak is None:
                            print(
                                f"{total_infers:>8}  {'--':>8}  {'--':>8}  "
                                f"{0.0:>12.5f}  {0.0:>10.4f} N  (no-contact)",
                                flush=True,
                            )
                        else:
                            vx, vy, vp = viz_peak
                            print(
                                f"{total_infers:>8}  {vx:>+8.2f}  {vy:>+8.2f}  "
                                f"{vp:>12.5f}  {fz_viz:>10.4f} N",
                                flush=True,
                            )

                last_window_seq = window_seq

            if has_gui and latest_viz_map_nmm2 is not None:
                should_render = viz_interval <= 0.0 or (now - last_viz_time) >= viz_interval
                if should_render:
                    if viz2d and viz3d:
                        if viz2d.query_xy_mm != query_xy:
                            query_xy = viz2d.query_xy_mm
                            viz3d.query_xy_mm = query_xy
                        elif viz3d.query_xy_mm != query_xy:
                            query_xy = viz3d.query_xy_mm
                            viz2d.query_xy_mm = query_xy

                    query_val_nmm2 = None
                    if query_xy is not None and latest_pred_map_nmm2 is not None:
                        query_val_nmm2 = engine.get_taxel_value(
                            latest_pred_map_nmm2,
                            query_xy[0],
                            query_xy[1],
                        )

                    if viz2d and open_2d:
                        viz2d.update(
                            latest_viz_map_nmm2,
                            peak=latest_viz_peak,
                            fz=latest_fz_viz,
                            query_val_nmm2=query_val_nmm2,
                        )
                    if viz3d and open_3d:
                        viz3d.update(
                            latest_viz_map_nmm2,
                            peak=latest_viz_peak,
                            fz=latest_fz_viz,
                            query_val_nmm2=query_val_nmm2,
                        )
                    last_viz_time = now
                    total_viz += 1
                    report_viz += 1
                    did_work = True

            if (now - last_report_time) >= args.report_interval_seconds:
                dt = max(now - last_report_time, 1e-6)
                data_hz = report_data_windows / dt
                infer_hz = report_infers / dt
                viz_hz = report_viz / dt
                print(
                    f"[perf] data={data_hz:6.1f}Hz  infer={infer_hz:6.1f}Hz  "
                    f"viz={viz_hz:6.1f}Hz  seq={last_window_seq}",
                    flush=True,
                )
                last_report_time = now
                report_data_windows = 0
                report_infers = 0
                report_viz = 0

            if has_gui:
                plt.pause(0.001)
            elif not did_work:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\nCtrl+C — 종료합니다.")
    finally:
        reader.stop()
        if has_gui:
            plt.close("all")
        print(
            "종료 완료. "
            f"data_windows={total_data_windows}, infers={total_infers}, viz_updates={total_viz}"
        )


if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    main()
