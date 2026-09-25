# Depth-Aware Contact Labeling & Training Checklist (2026-04-09)

목표: 압입 깊이(δ)에 따른 접촉 반경 확장을 라벨에 반영해 x,y 추정 안정성을 높이되, 기존 학습 구조를 그대로 유지할 수 있는 선택적 옵션으로 추가한다.

## 제약 및 원칙
- 기존 파이프라인/스크립트는 그대로 동작해야 함: 기본 설정은 기존 포인트 라벨/모델을 사용하도록 유지하고, 새 기능은 설정 플래그로 분기.
- 새 폴더/파일 생성은 허용하되 기존 파일을 파괴적으로 변경하지 않음; 공용 함수는 util 계층에 추가.
- 학습/추론 코드에서 깊이 채널이 없을 때도 실행 가능해야 함(깊이 미측정 시 고정 σ 사용).

## 생성·수정 대상(제안)
- 새 문서: `md/260409_depth_aware_contact_labeling.md` (본 문서)
- 새 모듈(선택): `training/utils/contact_geometry.py`에 접촉 반경/커널 생성 함수 추가
- 설정 예시: `training/configs/*` 또는 기존 argparse에 `--use_depth_aware_label`, `--depth_sigma_scale` 등 옵션 추가 (기존 default=False)

## 실행 체크리스트
1) 데이터/라벨 준비 (완료)
   - [x] 인덴터 반경 R=2.5 mm 확인 (지름 5 mm) 및 깊이 δ(1.0, 1.5 mm) 분포 점검
   - [x] 선택할 반경 모델 결정: `a = sqrt(R*δ)`(Hertz) 또는 `a = sqrt(2Rδ - δ^2)`(기하)
   - [x] 그리드 스텝(mm/셀) 계산 → 반경 a를 셀 단위로 변환
   - [x] 커널 선택 및 파라미터 σ 설정: Gaussian `w=exp(-d^2/(2a^2))` 또는 선형 `w=max(0,1-d/a)`
   - [x] 깊이 없는 샘플 처리 규칙 정의: a=const 또는 δ 평균값 사용
   - [x] 라벨 생성 스크립트/함수 작성 & 샘플 시각화로 검증 (preprocessing/ 완료)

2) 모델/손실 반영
   - [ ] 기존 모델 유지: 기본 헤드/손실은 변동 없음
  - [ ] 옵션 A: 좌표 회귀 보조로 2D heatmap 회귀 헤드 추가 (BCEwithLogits 또는 MSE)
  - [x] 옵션 B: 기존 heatmap 분류라면 소프트 타겟 허용하도록 손실만 수정 → `train_comparison.py`에 depth-aware soft heatmap + z/fz 보조 헤드 적용 (multi_head_field, 플래그 기반)
  - [x] 멀티태스크 시 z/힘 채널 또는 z-head 추가 여부 결정 (기본 off → 플래그 on 시 z/fz head 학습)

### 모델/손실 확정안 (옵션 B + z/fz 보조 헤드)
- 출력: `xy_heatmap [B,1,H,W] logits`, `z_depth [B,1]`, `fz [B,1]`
- 디코드: xy는 soft-argmax(또는 argmax+subpixel), z/fz는 linear
- 손실: `L_xy=BCEWithLogits(soft target, fg_weight≈3-10)`, `L_z=Huber`, `L_fz=Huber`
- 총합: `L = 1.0*L_xy + 0.2*L_z + 0.2*L_fz` (z/fz는 표준화 후 δ=1.0 사용)

### 아웃풋 스펙 표
```
Head        Shape          Target            Decode             Note
xy_heatmap  [B,1,H,W]      soft depth map    softargmax/argmax  weighted BCE first, alt: weighted MSE
z_depth     [B,1]          z_depth (norm)    linear             Huber δ≈1.0
fz          [B,1]          fz (norm)         linear             Huber δ≈1.0
```

3) 학습 파이프라인 변경
   - [x] argparse/config에 `--use_depth_aware_label` 플래그 추가 (default False)
 - [x] dataloader가 깊이 값을 batch에 포함하도록 확장 (없으면 None 처리) — 기존 Zarr/CSV tgt에 depth 포함, 그대로 활용
 - [x] 라벨 생성 모듈 연결: 플래그가 True일 때만 적용 (train_comparison.py, multi_head_field 전용)
 - [x] 로그/체크포인트 이름에 플래그 반영하여 기존 실험과 분리 (ckpt 파일명에 태그 추가, 로그 디렉토리명 미적용)
  - [x] 새 플래그: `--use_depth_aware_label`, `--depth_label_kernel gaussian`, `--depth_radius_model hertz|geom`, `--loss_xy bce|wmse`, `--loss_z huber`, `--loss_fz huber`, `--lambda_xy 1.0 --lambda_z 0.2 --lambda_fz 0.2`, `--decode_xy softargmax`

4) 실험 설계 및 검증
 - [ ] A/B 테스트: 기존 라벨 vs 깊이 의존 라벨 (같은 모델)
 - [x] 깊이 구간별 지표 보고: MAE/RMSE, 성공률(≤1 cell), 깊이별 분리
 - [x] 히트맵 품질 확인: 예측-정답 overlay 시각화
 - [ ] 데이터 순서 무작위화 및 드리프트 보정 여부 확인
  - [ ] ablation 순서: (1) point label + xy only → (2) depth-aware soft label + xy → (3) soft label + z/fz heads → (4) 필요 시 depth/force 입력 conditioning

5) 배포/호환성 확인
   - [x] 플래그 off일 때 모든 기존 학습/추론 스크립트가 이전과 동일한 결과를 내는지 스모크 테스트
   - [ ] 새 모듈 import 실패 시 graceful fallback (try/except 또는 조건부 경로)
   - [ ] 문서/README에 옵션 설명 추가

## 접촉 반경/커널 정의(초안)
```python
# R: mm, depth: mm, grid_res: mm_per_cell
import math

def contact_radius(depth, R=2.5, model="hertz"):
    if depth <= 0:
        return 0.0
    if model == "hertz":
        return math.sqrt(R * depth)
    return math.sqrt(max(0.0, 2 * R * depth - depth * depth))

def radial_weight(dist, a, kernel="gaussian"):
    if a <= 0:
        return 0.0
    if kernel == "gaussian":
        return math.exp(-dist * dist / (2 * a * a))
    return max(0.0, 1 - dist / a)
```

## 제안 실험 셋업
- 베이스라인: 기존 포인트 라벨, 동일 모델/하이퍼파라미터.
- 실험1: 깊이 의존 Gaussian 라벨, σ=a, `use_depth_aware_label=True`.
- 실험2: 실험1 + z/힘 채널 입력 또는 z-head 멀티태스크.
- 증강: δ를 0.8–1.7 mm 범위로 샘플링한 가상 라벨 생성으로 일반화 확인.

## 리스크/주의
- σ를 과대 설정하면 위치 분해능이 떨어짐 → 셀 단위 반경을 2–3칸 이내로 제한 권장.
- 깊이 값 미측정 데이터가 섞이면 라벨 스케일 불일치 발생 → 규칙 명시.
- 드리프트·히스테리시스는 라벨 변경만으로 해소되지 않으므로 데이터 수집 순서/보정 병행.

## 다음 액션 제안
1) 학습 argparse/config에 위 플래그 추가, 플래그 off 스모크 통과 확인
2) heatmap decode를 softargmax/argmax+subpixel로 고정하고 로깅 추가
3) Stage 2 실험: depth-aware soft label + xy only → Stage 3: + z/fz heads (λ=1/0.2/0.2)
4) z/fz 정규화 파라미터를 preprocessing 메타데이터에 저장 및 로드 경로 확인
5) 결과에 따라 λ_z/λ_fz, σ/a 스케일 재조정 및 ablation 표에 기록
