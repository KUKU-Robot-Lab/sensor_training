# 밴딩 P1~P3 합성 리허설 리포트

> 생성: `scripts/rehearse_bending_pipeline.py` (seed 7) — 취득 전 배관·인터페이스 검증.
> 합성 모델: Δs_i = k_i·(deg/90)·(z_i/9.75)·0.5·σ_signal, ±40°, 오프셋은 SATS 정규화 입력 공간에 주입(물리 충실도 아님 — 배관 검증 목적).

- frozen SATS: `ecomesh_xy0p5_sizeinput_val_d5t10_d10t3` (use_size_input=True), device=cuda
- 실 val 윈도우 8192개 (shape (8192, 10, 16)), 신호 std=9.4520
- **P1 estimator**: 600 step 학습 → 밴딩-only deg MAE = **0.77°** (±40° 범위)
- **P2 restorer(오프셋 지도)**: 오프셋 RMS 1.0211 → 복원 잔차 RMS **0.0151** (제거율 98.5%)
- **P3 pipeline(동결 SATS + size 전달)**: SATS 출력 RMSE vs flat 기준 — 보정 전 **0.7306** → 보정 후 **0.0162** (회복률 97.8%, 기준 출력 RMS 0.4496)
- P3 부수 관찰: 밴딩+접촉 혼합 신호에서 estimator deg MAE = 16.77° (밴딩-only 0.77° 대비 — 접촉 중첩 시 저하 정도 = 실데이터에서 검증할 리스크)
- **gradient 검증**: 동결 SATS 통과 backward → restorer grad 흐름 OK, SATS grad 없음(동결 유지) → **end-to-end(B안) 학습 가능 확인**. 단, **cudnn 제약 발견**: eval 모드 LSTM 은 backward 불가 → B안 학습 루프는 `torch.backends.cudnn.flags(enabled=False)` 로 감싸야 함(속도 저하 감수)

## 발견/조치
- `BendingPipeline`이 동결 SATS에 **size(인덴터 지름) 미전달** → A 모델에서 FiLM 누락 버그. forward 에 `size` 인자 추가 + A 모델에서 누락 시 명시 에러로 수정(이 리허설에서 발견).

## 실데이터에서 확인할 리스크 (리허설로 대체 불가)
- 밴딩+접촉 중첩 시 deg 추정 저하 폭 (위 P3 부수 관찰 항목의 실측판)
- 실제 오프셋의 z_i 선형성·k_i 안정성 (§5.3 가정)
- 취득 신호의 정규화: bending npz 는 **SATS 학습과 동일한 정규화 공간**으로 변환 후 입력해야 함