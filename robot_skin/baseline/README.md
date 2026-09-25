# baseline/ — 무접촉 ΔS 예측기

**`deformable_sats/sats/bending/` 의 deg→offset `BaselineRestorer` 를 일반화한 것.**
bending 은 "밴딩 각도 1개 → 16채널 오프셋"이었다면, 여기서는 손이 움직일 때 각 taxel 이
**접촉 없이도** 보이는 ΔS%(늘어남·굽힘·공압 커플링)를 `[taxel pose(pos, normal), q, q̇]` 로 예측한다.
접촉 신호는 `contact/` 에서 `residual = 관측 − 예측` 으로 분리한다.

| 파일 | 역할 |
|---|---|
| `model.py` | `BaselinePredictor`: taxel 공유 MLP + per-taxel 임베딩(게인/오프셋 흡수), 마지막 층 zero-init(항등 웜스타트 — restorer 와 동일) |
| `dataset.py` | `NoContactSession`(정렬된 ΔS·pose·q·q̇) → `NoContactWindowDataset`(윈도 끝 상태 → 윈도 평균 ΔS, 포화 윈도 제외, q/q̇ `NormStats`) |
| `train.py` | `train_baseline`(Huber, tail split — 윈도 겹침 누수 방지), `predict_session` |

bending 과 같은 원칙: 입력에 **관측 ΔS 를 넣지 않는다**(넣으면 접촉까지 오프셋으로 학습해 지운다 —
`eval_contact_preservation` 에서 확인된 seq_deg 붕괴).
