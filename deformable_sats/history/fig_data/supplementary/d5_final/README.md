# D5 final — d5-only + β 논문 스타일 SATS 재현 · 해석 가이드

**모델**: d5-only + β(물성보정), 출력 0.5mm(41×41). run `d5only_beta_g0p5`. 순수 SATS 구조(크기입력 없음). 논문 Fig4 스타일 패널.

## 패널별 해석

### `D5A_lineprofile_ecomesh.png` — 중앙선 압력 프로파일 (SATS 핵심)
- 대칭선을 따라 force별로 추론한 압력 단면. 16 물리 taxel에서 연속 압력 분포를 복원 = super-resolution의 시각적 증거.
- **읽는 법**: force가 클수록 peak 높음. 부드러운 곡선 = 조밀 가상 taxel 복원 성공. 물리 taxel 위치(성긴)와 대비.

### `D5C_pressure3d_ecomesh.png` — 3D 압력 맵
- 단일 접촉의 조밀 가상 taxel 압력 분포(3D surface).
- **읽는 법**: 단일 봉우리 형태·위치가 접촉과 일치하면 정확. 격자 조밀도 = SR 배율.

### `D5D_poserror3d_ecomesh.png` — 위치별 오차 3D
- 표면 위치별 위치추정 오차(bar3d). 낮고 균일할수록 좋음.
- **읽는 법**: 특정 위치만 튀면 그 영역 취약. 전반 낮으면 표면 전체 정확.

### `D5E_error_hist_ecomesh.png` — 오차 히스토그램
- per-sample 상대오차 분포. 왼쪽(0 근처)에 몰릴수록 좋음.
- **읽는 법**: 분포 중앙값·꼬리 확인. 긴 오른쪽 꼬리 = 일부 큰 오차 샘플.

### `D5F_force_error_ecomesh.png` — force별 오차
- fz 전 구간 상대오차. 힘에 걸친 안정성.
- **읽는 법**: 평평하면 force 무관 안정. 저force에서 상승은 분모(작은 target) 효과.

## 축 통일
- `--ref-limits`로 shared_axes/axis_limits.json 주입 → 소재 figset과 동일 축(비교 가능).

## 재현
```bash
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_fig3_sats.py \
  --figset d5_final --ref-limits "history/fig_data/fig2_material_ablation/panelD_sats/xy1_material/shared_axes/axis_limits.json"
```
(B 패널=소재 비교는 d5_final이 단일 소재라 자동 skip)

## 관련
- 다해상도(1.0/0.25/0.1mm) 및 GT vs Pred 갤러리·해상도 비교 = `experiments_archive/d5_multires_diag/README.md` 참조.
