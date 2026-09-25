# FigS29 — self-attention 해석성 (t-SNE)

> **2026-07-08 A 갱신**: **A(크기 입력) 모델**(`runs/size_input_material/sizeA_*`, final=`runs/size_input/`)로 재생성. attention 특징은 FiLM(크기 조건) 이전 단계라 결과는 크기 입력과 무관 → 해석·논지 동일.

논문 FigS29 대응. 한 감지 유닛(node)의 인코딩 feature를 **self-attention 모듈 전/후**로 각각
t-SNE 2D 축소하고 press 위치로 색을 매핑한다.

## 판정 방법
- attention 전 = `model.encoder(...)` 출력, attention 후 = `model.attention(local_feat)` 출력. (feature dim 64)
- 활성 node = attention 전 feature 노름 분산 최대 유닛 자동 선택. 해당 node 강응답(근처 press) 표본만 t-SNE.
- 색 = press 위치(x,y) → 2D RGB (가까운 위치 = 가까운 색).

## 결과 (논문 논지 재현)
- **before attention**: feature가 1D 곡선형 매니폴드 — 위치 색이 뒤섞임 → **거리(자극 크기) 위주**, 위치 미분리.
- **after attention**: 2D 공간 구조로 정렬 — press 위치 색이 매끄러운 gradient → **self-attention 이 공간정보를 디커플링**.
- 전 소재/해상도 모델에서 동일 경향 → SATS 구조(attention 기반 다중유닛 정보 집계)의 타당성 방증.

## 코드 (재현)
```bash
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_supp_attention_tsne.py \
    --models eco20_xy1 eco50_xy1 ecomesh_xy1 ecomesh_xy0p5_final
```
스크립트: `history/fig_data/visualizing_scripts/figure_set/generate_supp_attention_tsne.py`
(`collect_features`=feature 추출·node선택, `plot_tsne`=t-SNE+위치색). 별도 npz 불필요.
