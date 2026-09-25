# contact/ — 잔차 → 접촉 해석 (전부 구현)

| 파일 | 역할 |
|---|---|
| `residual.py` | `residual = 관측 ΔS − baseline 예측`, `contact_mask`(−residual ≥ 임계 & trusted) |
| `ordinal.py` | `OrdinalQuantizer`: {0 무접촉, 1 약, 2 강, 3 포화}, `one_hot` |
| `saturation_fsm.py` | per-taxel `SaturationFSM`: OK → SATURATED → RECOVERING → OK(±ok_pct 로 ok_sec 안정) / 타임아웃 시 **re-zero**(offset ← 현재 잔차). 기본값은 `sats/inference/run_dashboard.py` 격리 규칙(1.5 %, 2 s, 30 s) |
| `self_touch.py` | 손가락 세그먼트(캡슐) 거리 기반 **자가 접촉 자동 라벨** — 자기 손가락 체인은 제외 |

부호: ΔS 는 SATS 규약(누르면 음수). 임계값은 `common.signal.press_intensity`(= −ΔS) 기준의 양수 %.
