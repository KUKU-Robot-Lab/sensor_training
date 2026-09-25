# representation/ — taxel 토큰

| 파일 | 상태 | 역할 |
|---|---|---|
| `tokenizer.py` | 구현 | `TaxelTokenizer`: 값(잔차/ordinal one-hot 등) MLP + pose(Fourier(pos) ⊕ normal) MLP (+ 선택적 id 임베딩) → LayerNorm 토큰 `[B,N,D]`. `mask` 로 값 부분만 `[MASK]` 치환 |
| `pretrain.py` | 스텁 + 헬퍼 | `random_taxel_mask` 구현, `MaskedTaxelPretrainer`(마스킹 재구성 사전학습) 스텁 |

pose 기반이라 taxel 수·배치가 다른 글러브/로봇 핸드가 같은 토크나이저를 공유할 수 있다.
