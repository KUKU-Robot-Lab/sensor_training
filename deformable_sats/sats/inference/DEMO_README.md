# v6 실시간 추론 데모 (학회 데모세션)

v6 학습 파일로 실시간 추론. **터미널 우선** + 2D/3D 시각화. 엔트리: `sats/inference/run_demo.py`.

## 모델·보정 자산
- SATS 배포: `sats/training/runs/ecomesh_v6_deploy_all4`
- 밴딩 estimator: **`sats/bending/runs/estimator_v6new/best.pt`** (신규 v6 y23-33 재취득, G1 MAE 1.78°·θ 0~157°; 없으면 구 `estimator_v6` 폴백) + `estimator_v6new/best.pt_ref_baseline.npy`
- 밴딩 학습데이터(restorer 온더플라이): `learning_data/bending/v6_new`(δ=y−23, 유효 0~10mm; 없으면 v6 폴백)
- z 보정 LUT: `sats/inference/z_calibration_v6.json` (맵 peak→z_depth, d5 R²0.87·d10 R²0.59, **근사값**)

## 0. 기본 셋업 / 포트 찾기 (실기 시작 전)

```bash
# 환경 확인
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# ① 센서 USB 연결 후 — 시리얼 포트 목록(device/description/hwid)
.venv/bin/python -m sats.inference.run_demo --list-ports
#   커널 인식 확인(센서 꽂은 직후): dmesg | grep -i tty | tail ; ls -l /dev/ttyUSB* /dev/ttyACM*

# ② 연결·데이터 흐름 확인(raw 바이트 수신 여부)
.venv/bin/python -m sats.inference.run_demo --probe --port /dev/ttyUSB0 --baudrate 250000

# 권한 오류(Permission denied) 시
sudo usermod -aG dialout $USER   # 재로그인 후 영구 적용
sudo chmod 666 /dev/ttyUSB0      # 또는 임시

# 하드웨어 없이 파이프라인만 점검
.venv/bin/python -m sats.inference.run_demo --mode contacts --mock
```

포트를 찾으면 아래 명령들의 `--port` 에 지정한다.

## 산출물 ↔ 실행

**환경**: `.venv/bin/python`. 시리얼 포트/보드레이트/protocol은 실센서에 맞게 조정.
`--list-ports`로 포트를 먼저 확인할 것.

```bash
# ① 단일접촉 x,y,z,fz (터미널)
.venv/bin/python -m sats.inference.run_demo --mode contacts --contacts 1 --diameter 5 --port /dev/ttyUSB0

# ① 다중접촉(최대 3) x,y,z,fz
.venv/bin/python -m sats.inference.run_demo --mode contacts --contacts 3 --diameter 10 --port /dev/ttyUSB0

# ② +시각화(2D/3D heatmap + 최적프레임). 위 명령에 --viz 추가
.venv/bin/python -m sats.inference.run_demo --mode contacts --contacts 3 --diameter 10 --viz 2d --port /dev/ttyUSB0
#   --viz {2d|3d|both}
#   --show-units : 실시간 16-taxel 센싱유닛 heatmap(SATS 입력 원본) 창 추가(밴딩/접촉 패턴 확인)

# ③ 밴딩각 theta (터미널)
.venv/bin/python -m sats.inference.run_demo --mode theta --port /dev/ttyUSB0

# ④ SATS+밴딩 통합 (theta: x,y,z,fz)
.venv/bin/python -m sats.inference.run_demo --mode bending --contacts 3 --diameter 10 --viz 2d --port /dev/ttyUSB0
```

- `--min-distance-mm`: 접촉 간 최소 간격. **미지정 시 `--diameter` 값으로 자동**(d5→5·d10→10mm).
  같은 지름 두 접촉은 중심간 지름보다 가까울 수 없으므로, 단일접촉이 인접 peak 2개로 쪼개지는 것을 막는다.
  진짜 두 접촉을 더 가깝게 붙여야 하면 이 값을 직접 낮춘다.
- `--mock`: 하드웨어 없이 파이프라인 확인(단 theta/bending은 flat baseline이 없어 비의미).
- `--report-interval`: 최적(최대 Fz) 프레임 요약 주기. `--infer-max-fps`/`--viz-fps`: 속도.

## 산출물별 동작
1. **contacts**: 매 프레임 접촉별 `x,y,z,fz` 출력 + 최대 Fz 프레임 latch. z=peak_val LUT 근사, fz=Voronoi 적분.
2. **viz**: 단일 패널 live heatmap(단색 초록, 상대 정규화), 접촉 마커(최대3)·라벨·theta.
3. **theta**: ★**재앵커**(학습 참조 baseline에 pct) → 기압/온도/유닛 무관. 시작 flat=theta0 영점 → Δθ.
   표시 방식 `--theta-mode`: **hold**(기본, 지그 고정각 peak-hold — 응력완화 creep 무시 상수) ·
   **dynamic**(밴딩/해제 제스처가 0으로 복귀, 동적 시연용) · **live**(순수 연속).
   ※ 지그에 물린 고정 밴딩은 응력완화로 연속값이 0으로 수렴 → **클램프 데모는 hold** 필수.
   estimator는 연속 램프 학습이라 정적 hold와 신호 불일치(그래서 peak 래치). 유효 밴딩 ~20-150°.
4. **bending 상태머신**:
   1. 시작 → **플랫 무접촉 baseline** 캡처(센서에서 손 떼고 대기).
   2. **지그 장착·밴딩** 후 **Enter** → 밴딩 무접촉서 theta 고정(재앵커).
   3. 접촉 press → `restorer(pct,theta)` 복원 → 동결 SATS → `theta: x,y,z,fz`.

## ★ 내일 실기 체크리스트
- [ ] serial `--protocol`(기본 auto→binary, vensor2.ino=바이너리)·`--baudrate`(250000)·`--port` 실센서 확인.
- [ ] baseline 캡처 시 **무접촉 유지**(플랫). d5/d10 `--diameter` 지정(z·size 조건).
- [ ] theta: **재앵커**라 기압 무관하나, `estimator_v6_ref_baseline.npy`가 estimator 옆에 있어야 함
      (없으면 데모 baseline 폴백=기압 민감). 시작 flat 영점(2초 무접촉·미밴딩 유지) → 이후 Δθ.
- [ ] theta 스케일: estimator는 **buckling(Y구동 δ=Y−18)** 학습 → 데모 지그도 **버클 방식**이어야 theta 유효. 순수 굽힘이면 정합 데이터 재학습 필요([[v6-test-eval]]).
- [ ] 다중접촉: 접촉을 **동등·firm하게** 눌러야 분리 선명(약접촉은 확산).
- [ ] z는 **근사**(특히 d10 R²0.59)임을 데모 설명에 명시.

## 신규 모듈
`z_calibration.py`(z LUT) · `demo_contacts.py`(추출·터미널·latch) · `bending_infer.py`(theta·복원) · `demo_viz.py`(2D/3D) · `run_demo.py`(엔트리). 테스트 `tests/test_demo_contacts.py`·`test_demo_viz.py`.


## S3 패널 — 변형 복원(각도-프리)

각도 추정 없이 **변형된 상태의 baseline 을 되돌려** SATS 입력을 학습 분포로 복구한다.

```
restored_pct = pct - offset(z(pct))
```

- 체크포인트: `sats/bending/runs/deform_restorer/best.pt` (`--deform-restorer` 로 변경)
- 있으면 **재장착([b]) 불필요** — 창마다 변형 오프셋을 다시 추정한다
- **파손 taxel 마스킹은 체크포인트에 저장된 목록을 따른다**(학습과 동일해야 함)

### 데모 진행 — 구분 동작

| 순서 | 조작 | 화면 |
|---|---|---|
| ① | 평평한 상태 | 맵 비어 있음 |
| ② | 손으로 변형 후 **정지** | `RESTORE OFF` 로 두면 유령이 뜬다 |
| ③ | **[r]** 또는 `복원` 버튼 | `RESTORE ON` — 유령이 사라진다 |
| ④ | 변형 유지한 채 터치 | 접촉만 표시 |

②↔③ 를 번갈아 눌러 before/after 를 보여주는 것이 가장 명확하다.

### 실측 (v7, 13채널 — S07·S11·S15 파손)

| 지표 | 값 |
|---|---|
| 유령 억제율 | **89.8%** (leave-one-session-out, 4세션) |
| 접촉 회복 | 65.5% (준-합성) |
| 변형이 더한 위치오차 | 약 1.4mm (기준선 2.13mm → 3.77mm) |

억제율은 온전한 16채널 v6(87%)과 동등하다. 남은 위치오차는 대부분 파손된 taxel 때문이며
변형 데이터를 더 모아도 개선되지 않는다.
