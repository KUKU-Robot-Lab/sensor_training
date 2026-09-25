# eval/ — 지표 (구현)

| 함수 | 정의 |
|---|---|
| `hallucination_rate` | 실제 무접촉 단위(frame 또는 taxel-frame) 중 접촉이라 예측한 비율 |
| `motion_contact_separability` | press intensity 로 "실접촉" vs "움직임만(무접촉)" 분리: AUROC·d′ — baseline 차감 전/후 비교가 핵심 |
| `saturation_recovery_times` | SATURATED 이탈 → OK 복귀까지 시간(s), 에피소드별 |
| `auroc` | Mann–Whitney (동률 ½) |

stage 들은 이 함수들로 `metrics.json` 을 채운다(baseline `sep_auroc_before/after`, contact `*_hallucination_*`,
`*_auroc` …). 키 목록과 읽는 법: `docs/TRAINING.md` §12. 합성 데이터에는 정답 기반 `gt_*` 지표가 추가된다.
