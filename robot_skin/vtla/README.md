# vtla/ — 촉각 → VLA(VTLA) 어댑터 (구현)

| 클래스 | 역할 |
|---|---|
| `TactileTokenAdapter` | taxel 토큰 `[B,N,d]` → 학습 쿼리 K개 cross-attention → `[B,K,d_out]` (taxel 수 무관) |
| `ContactGate` | `hard`: 접촉 없으면 0 / `soft`: 접촉 시 σ(w·접촉비율+b) — 무접촉 시 여전히 0 |

목적: 무접촉 구간에서 촉각 스트림이 백본에 아무것도 주지 않게 해서 드리프트로 인한 **촉각 환각**을 구조적으로 차단.
