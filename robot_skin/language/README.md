# language/ — 작업 지시문 → 언어 토큰 (VTLA 의 L)

지시문은 D2 `task` 에피소드마다 하나 (`episode.meta.instruction`: 작업 카탈로그 템플릿으로 생성,
작업자가 `events.jsonl` 의 `instruction` 이벤트나 `manifest.task.instruction` 으로 덮어쓸 수 있음).

## 계약 (`TextEncoder`)

```python
batch = enc.tokenize(texts)                 # {"input_ids": long[B,L], "pad_mask": bool[B,L]} — CPU/파이썬
tokens = enc(batch["input_ids"], batch["pad_mask"])   # float[B,L,D], 패딩 위치는 0
tokens, pad_mask = enc.encode(texts)        # 두 단계 한 번에 (pad_mask True = 패딩)
```

`tokenize` 는 DataLoader 워커/collate 에서, `forward` 는 모델 안에서 (DDP·torch.compile 친화적) 돌릴 수 있다.
`enc.get_tokenizer()` 는 가중치 없는 피클 가능 콜러블(`HashingTokenizer` / `HFTokenizerFn`)을 돌려주므로
데이터셋·collate 는 인코더 모듈 대신 이것만 들고 있으면 된다 (`HashingTokenizer(max_len, vocab_size, hash_seed=…)`
를 직접 만들어도 같은 id).
`fixed_len` 이 정수면 모든 배치가 같은 L (정적 shape), `None` 이면 배치 최장 길이.

| `type` | 클래스 | 의존성 | 비고 |
|---|---|---|---|
| `hashing` (기본) | `HashingTextEncoder` | 없음 | 학습 가능, 결정적 해싱 |
| `hf` / `clip` / `siglip` / `t5` | `HFTextEncoder` | transformers | 사전학습 텍스트 타워 (기본 동결) + 선택 투영 |

### HashingTextEncoder

- 토크나이저: NFKC 정규화 → casefold → 유니코드 단어(`\w+`) 분리, 구두점 제거. 한국어 어절도 그대로 한 토큰
  (`"컵을 들어"` → `["컵을", "들어"]`; 조사까지 한 단어로 해싱되므로 한국어 지시문이 많으면 HF 다국어 인코더를 고려).
- id = `2 + crc32(f"{hash_seed}:{word}") mod (vocab_size − 2)` — 어휘 파일 없음, 프로세스·머신 간 동일
  (Python `hash()` 와 달리 시드 무작위화 없음). `PAD_ID=0`, `BOS_ID=1`.
- 위치 0 은 항상 `[BOS]` (요약 토큰) → 빈 지시문도 유효 토큰 1개 (완전 마스킹 행 → attention NaN 방지).
  `max_len` 은 BOS 포함, 초과 단어는 잘림.
- 임베딩: `nn.Embedding(vocab_size, dim, padding_idx=0)` + 학습 위치 임베딩 + LayerNorm, 선택적으로
  pre-LN 트랜스포머 `n_layers` 층 (패딩 마스크 적용 → 패딩 길이와 무관한 출력).
- 지시문이 소수의 템플릿에서 나오는 D2 에는 이 정도로 충분하고 빠르다.

### HFTextEncoder

- `AutoTokenizer` + `AutoModel`; 이중 인코더(`CLIPModel`, `SiglipModel`)는 `text_model`, 인코더-디코더(T5)는 encoder 만 사용.
- 패밀리 기본값: `clip` → `openai/clip-vit-base-patch32` (max_len 77), `siglip` → `google/siglip-base-patch16-224`
  (max_len 64, `padding: max_length` — SigLIP 학습 방식), `t5` → `t5-small`.
- 토크나이저가 attention mask 를 만들지 않으면(SigLIP) 모든 위치를 유효로 본다. 빈 문자열로 전부 패딩이 되는
  행은 위치 0 을 유효로 바꿔 완전 마스킹을 막는다. 패딩은 항상 오른쪽 (`padding_side="right"` 로 고정 →
  유효 토큰이 앞쪽 prefix, `InstructionCache` 와 같은 배치).
- `frozen=True` (기본): 백본 동결·eval 고정, `out_dim` 투영층만 학습. 동결 여부는 `requires_grad` 에서 매번 읽으므로
  백본 파라미터를 직접 `requires_grad_(True)` 하면 미세조정이 된다.

## InstructionCache

동결 인코더의 출력을 지시문 문자열별로 메모이즈 (에피소드당 지시문 1개, 데이터셋 전체에서도 소수).

```python
enc = build_text_encoder({"type": "clip"})          # 동결
cache = InstructionCache(enc)
cache.warm({ep.meta.instruction for ep in episodes})  # 한 번에 인코딩
tokens, pad_mask = cache.encode(batch["instruction"], device="cuda")   # encoder.encode 와 같은 계약
cache.save(run_dir / "instructions.pt")              # GPU 박스에서 미리 계산 → 다른 머신에서 load
```

- 유효 토큰만 저장(CPU, `dtype` 선택)하고 `encode` 때 `fixed_len` 또는 배치 최장 길이로 다시 패딩.
- 학습 가능한 인코더는 거부 (`allow_trainable=True` 로 무시 가능) — 첫 옵티마이저 스텝 후 캐시가 낡기 때문.
- `load` 는 인코더 `cache_key` 가 다르면 거부 (`strict=False` 로 무시). `cache_key` 는 모델/토크나이즈 설정
  (HF: 모델 id·revision·scratch, hashing: vocab·seed·dim·max_len·대소문자)을 식별할 뿐 학습된 가중치는 구분하지
  못한다 → 학습 후 동결한 인코더는 체크포인트마다 별도 캐시 파일을 쓸 것.

## 설정 예

```yaml
language:
  type: hashing        # hashing | clip | siglip | t5 | hf
  dim: 256
  max_len: 32
  n_layers: 0
```

선택 의존성이 없으면 `ImportError` 에 설치 방법과 대안(`type: hashing`)이 표시된다. 테스트는 선택 의존성 없이 돈다.
