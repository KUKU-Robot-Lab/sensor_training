# Fig.4 — Application (논문 §6 Fig.4)

> 상태: **미착수** (하드웨어 데모 취득 전제).

## 계획 (논문 §6·§7)

1. 로봇핸드 부착 실시간 접촉 감지·제어 데모
2. 사람 손 곡면 부착 동작 시연

## 필요 인프라

- 실시간 추론: `sats/inference/` (realtime) + 밴딩 프론트엔드 `sats/bending/pipeline.py`
- 로깅: `skin_ws/acquisition_code/final_logger_integrated_v3_gui`
