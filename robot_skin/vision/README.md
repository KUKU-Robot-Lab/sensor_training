# vision/ — 카메라 프레임 → 시각 토큰 (VTLA 의 V)

```
camera_<name>/frames.npy uint8[F,H,W,3]
   │  transforms (TrainAugment | EvalTransform)  → float[B,3,h,w], encoder.mean/std 로 정규화
   ▼
VisionEncoder.forward  → tokens [B,P,D]   (P = encoder.n_tokens(h,w), D = encoder.out_dim)
   │  (frozen 인코더면) feature_cache → derived/vision_<key>_<camera>.npy  float16[F,P,D]
   ▼
vtla.VTLAPolicy (카메라 임베딩 + 융합 트랜스포머)
```

| 모듈 | 내용 |
|---|---|
| `encoders.py` | `VisionEncoder` 기반 클래스, `TinyConvEncoder`, `ResNetEncoder`(torchvision), `HFVisionEncoder`(transformers), `TokenPool`, `SpatialSoftmax`, `sincos_pos_embed_2d`, `build_vision_encoder(cfg)` |
| `transforms.py` | `to_float_tensor`, `Normalize`, `resize`/`resize_short`/`center_crop`, `crop_resize`, `TrainAugment`, `EvalTransform`, `build_transforms(cfg, encoder)` |
| `feature_cache.py` | `cache_episode_features`, `cache_features`, `load_cached`, `has_cached`, `gather_frame_features`, CLI |

## 인코더 계약

- 입력: `float [B,3,H,W]`, **`encoder.mean`/`encoder.std` 로 정규화된** 이미지 (uint8 프레임은 transforms 로 변환).
- 출력: 토큰 `[B,P,D]`. `out_dim = D`, `n_tokens(h,w) = P` (Tiny/ResNet 은 해석적 계산, HF 는 더미 forward 1회 후 캐시).
- `cache_key`: 캐시 파일 이름용 문자열. `is_frozen`: 학습 가능한 파라미터가 하나도 없는지 (캐시 가능 여부).
- `freeze()`: 전체 동결 + eval.

| `type` | 클래스 | 의존성 | 토큰 | 용도 |
|---|---|---|---|---|
| `tiny` (기본) | `TinyConvEncoder` | 없음 | `grid` 기본 4×4 = 16개 (이미지 크기 무관) | 테스트, 스모크, 스크래치 학습 |
| `resnet18/34/50`, `resnet` | `ResNetEncoder` | torchvision | stride 32 셀마다 1개 (`pool=none`) | 사전학습 CNN, Diffusion Policy 식 설정 |
| `hf`, `dinov2`, `siglip`, `clip` | `HFVisionEncoder` | transformers | 패치 토큰 (+CLS) 또는 `tokens=pooled` 1개 | 강한 사전학습 ViT (고정 → 캐시) |

`pool` (Tiny/ResNet): `none` (셀마다 토큰) · `grid` (적응 평균 풀링으로 고정 격자) · `avg` (1개) ·
`spatial_softmax` (K개 키포인트 토큰 = 기대 좌표 (x,y) + 주의 가중 특징). `grid`/`none` 토큰에는 MAE
(He et al., arXiv:2111.06377)식 고정 2-D sin-cos 위치 임베딩을 더한다.

`frozen=True` 는 **백본만** 동결한다 (`out_dim` 을 주면 투영층은 학습됨). 동결 ResNet 은 `train()` 중에도
BatchNorm 통계가 변하지 않도록 백본을 eval 로 유지한다. 캐시를 쓸 때는 `frozen=True, out_dim=None`
(→ `is_frozen == True`) 으로 두고, VTLA 모델 쪽 투영층이 학습되게 한다.
동결 여부(`enc.frozen_backbone`)는 파라미터의 `requires_grad` 에서 매번 읽는다 → 미세조정 시
`for p in enc.backbone.parameters(): p.requires_grad_(True)` 만 하면 gradient 가 흐르고, 다음 `.train()` 부터
BN/dropout 도 학습 모드가 된다. `tiny` 의 `out_dim: null` = 마지막 stage 폭 그대로 (투영 없음).

설계 근거: Diffusion Policy (Chi et al., arXiv:2303.04137) — ResNet-18 + BatchNorm→GroupNorm + spatial
softmax 풀링, 작은 random crop. OpenVLA (arXiv:2406.09246) — DINOv2 + SigLIP 비전 타워 조합.
두 가지 모두 설정으로 선택 가능 (`group_norm: true`, `pool: spatial_softmax`, `type: dinov2|siglip`).

```yaml
vision:
  type: resnet18          # tiny | resnet18 | dinov2 | siglip | clip | hf
  pretrained: true
  frozen: true
  out_dim: null           # null = 백본 폭 그대로 (캐시 가능)
  pool: none
image:                    # build_transforms
  image_size: [224, 224]
  scale: [0.8, 1.0]       # random resized crop 면적 비율
  brightness: 0.2
  contrast: 0.2
  seed: null              # null = torch 전역 RNG (DataLoader 워커별 시드)
```

## transforms

- `to_float_tensor`: uint8 `[...,H,W,3]` (numpy/torch, 읽기 전용 memmap 포함) → float32 `[...,3,H,W]` ∈ [0,1]. 앞쪽 차원 보존.
- `EvalTransform(out_size, crop_scale=1)`: 중앙 crop (출력 종횡비, 면적 비율 `crop_scale`) → 리사이즈 → 정규화. 결정적.
- `scale`/`crop_scale` 은 **출력 종횡비를 가진 최대 crop 대비 면적 비율**이다 (전체 이미지 면적 대비가 아님).
  예: 640×480 카메라 → 224×224 출력이면 최대 crop 은 480×480, `scale=0.81` 은 432×432. 그래서 카메라와 출력의
  종횡비가 달라도 `scale=(0.8,1)` 줌 증강이 실제로 걸린다. 축소가 2배를 넘으면 축마다 anti-alias 프리필터를 쓴다.
- `TrainAugment(out_size, scale=(0.8,1), ratio=(1,1), brightness, contrast, saturation, seed)`:
  샘플별 random resized crop (`affine_grid`+`grid_sample`, 크게 축소할 때는 anti-alias 프리필터) + 색 지터 + 정규화.
  파라미터는 **첫 번째 차원별**로 뽑고 나머지 앞쪽 차원에 공유 → `[B,T,3,H,W]` 관측 히스토리는 같은 crop.
  좌우 뒤집기는 없음 (손·장면 기하와 행동 좌표가 뒤집히므로).
- `TrainAugment.eval_transform()`: 평균 학습 crop 면적으로 중앙 crop → 학습/평가 줌 일치. `build_transforms` 기본값.
- 난수: `seed=None` → torch 전역 RNG (DataLoader 가 워커·에폭마다 시드, `seed_everything` 으로 재현).
  `seed=int` → 전용 `torch.Generator`; DataLoader 워커 안에서는 `(seed, worker seed)` 로 워커·에폭마다 한 번
  재시드해서 워커끼리 같은 증강을 반복하지 않는다. 피클 가능 (spawn 워커).
- 모든 연산은 입력 텐서의 디바이스에서 실행 → GPU 배치 증강도 가능.

## feature cache (동결 인코더 가속)

```python
from robot_skin.vision import build_vision_encoder, EvalTransform, cache_episode_features, load_cached
enc = build_vision_encoder({"type": "dinov2"}).freeze()
tf = EvalTransform(224, mean=enc.mean, std=enc.std)
cache_episode_features(ep, "ego", enc, tf, device="cuda", batch_size=128, autocast_dtype=torch.bfloat16)
feats = load_cached(ep, "ego", enc.cache_key)            # float16 [F,P,D] (mmap)
x, valid = gather_frame_features(feats, ep[cam_idx_key("ego")][ticks])   # 마스터 틱 → 프레임
```

- 파일: `<episode>/derived/vision_<key>_<camera>.npy` (+ `.json` 사이드카: 인코더, transform, 크기, 생성 시각).
  원자적 쓰기 (tmp → rename), float16 범위 초과 값은 클립 + 경고.
- **프레임 단위** `[F,P,D]` (마스터 클럭 200 Hz 가 아님). `Episode.set_derived`(T 행 요구)를 쓰지 않으므로
  `Episode.derived()` 가 아니라 `load_cached()` 로 읽는다. 재전처리로 프레임 수가 바뀌면 `load_cached` 가 stale 로 거부.
- 이미 있으면 건너뜀 (`overwrite=True` 로 재계산). 단 사이드카에 기록된 인코더 `cache_key`·`out_dim`·eval transform
  이 지금과 다르면 `ValueError` (다른 파이프라인의 특징을 조용히 재사용하지 않음; `overwrite=True` 또는 다른 `key`).
  학습 가능한 파라미터가 있는 인코더는 경고 (캐시가 학습을 못 따라감).
- 검증에는 `camera_<name>/timestamps.npy` 또는 사이드카만 필요 → 학습 박스로는 `derived/` + `timestamps.npy`
  만 복사해도 된다 (프레임 불필요).
- VTLA 학습(캐시)과 배포(온라인 인코딩)는 **같은 eval transform** 을 써야 한다: 사이드카 `transform` 에 기록된
  `EvalTransform(...)` 과 `build_transforms` 의 eval 이 같은지 확인 (CLI 기본 `--crop-scale 1.0`,
  `build_transforms` 기본 eval crop 은 학습 scale 평균).
- 증강은 캐시와 양립하지 않는다 (동결 특징 사용 시의 일반적인 절충).
- CLI (GPU 박스에서 인코더별 1회):
  `python -m robot_skin.vision.feature_cache --root robot_skin/data/processed --dataset task --encoder '{"type": "resnet18"}' --image-size 224 224 --device cuda --bf16`

## GPU 메모 (RTX 5090 등)

- 선택 의존성은 torch 빌드와 맞춰 설치: Blackwell(sm_120)은 CUDA ≥ 12.8 휠 — `pip install torchvision --index-url https://download.pytorch.org/whl/cu128`.
  `transformers` 가중치는 `HF_HOME` 캐시를 공유하면 여러 머신(Tailscale)에서 재다운로드를 피할 수 있다.
- VTLA 학습용 캐시는 `vtla` stage 가 직접 만든다(`--set vision.cache_features=true`: 인코더 동결, 키 =
  `cache_key` + 인코더 가중치 해시, eval 변환) → 이미지 디코딩/인코딩 비용 제거. 위 CLI 로 만든 캐시는 키에 가중치
  해시가 없어 stage 가 재사용하지 않는다 — 분석·다른 모델용 (`docs/TRAINING.md` §13).
- 온라인 인코더 학습 시에는 `TrainAugment` 를 GPU 텐서에 적용해도 된다 (배치 연산).
- 선택 의존성이 없으면 `ImportError` 에 설치 방법과 대안(`type: tiny`)이 표시된다. 테스트는 선택 의존성 없이 돈다.
