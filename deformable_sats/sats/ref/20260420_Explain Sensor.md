## Specification of Skin Sensor

FPCB 회로판에 BMP384(Bosh) 16개가  4x4 array, 센서 간의 간격이 센서의 중심 기준으로 6.5mm 간격으로 배치됨.
BMP384 SPI 통신을 사용하며, Arduino DUE와 연결되고, DUE를 통해 컴퓨터로 raw count 값을 출력함.
FPCB와 BMP384 전체에 ECO20(Soomth-on)을 부어 경화시킴.

전체 사이즈는 25x25x3
Sensing Area : x=[-9.75, 9.75], y=[-9.75, 9.75] 

##Sensor 좌표 : 26.04 기준
|       | **S4** | **S3** | **S2** | **S1** |
|   --- |    --- |    --- |    --- |    --- |
| **x** | -97500 | -32500 | 32500 | 97500   |
| **y** | -97500 | -97500 | -97500 | -97500 |
|       | **S8** | **S7** | **S6** | **S5** |
| **x** | -97500 | -32500 | 32500 | 97500   |
| **y** | -32500 | -32500 | -32500 | -32500 |
|       | **S12** | **S11** | **S10** | **S9** |
| **x** | -97500 | -32500 | 32500 | 97500 |
| **y** | 32500  | 32500  | 32500 | 32500 |
|       | **S16** | **S15** | **S14** | **S13** |
| **x** | -97500 | -32500 | 32500 | 97500 |
| **y** | -97500 | -32500 | 32500 | 97500 |

##Super-resolution tactile sensor arrays with sparse units enabled by deep learning
- 논문 학습구조(SATS)을 기반으로 하는 센서 개발.

##데이터 취득 방법 for sats
1) 3-AXIS 스테이지 모터(0.1 micro-meter 단위 움직임 가능)를 이용하여 s1부터 s16까지 압입 실험 진행.
2) 0.5mm 단위로 x 또는 y가 움직이면서 Grid point에서 압입을 진행함 (40x40 Grid)
3) z는 최대 1.0mm 또는 1.5mm 까지 압입함.
4) x,y=(9.75, -9.75)위치에서 시작하여 x를 0.5mm씩 증가시키면서 압입하고, x의 센싱 레인지의 끝에 도달했을 때, y를 0.5mm 증가심. 즉, 지그재그로 데이터를 취득하고 모든 grid cell 압입을 완료할때까지 진행함.
4) 데이터는 vensor(200hz), FT-Sensor(100hz), motor motion을 전부 기록 하여 raw_merge.py를 통하여, 근처값을 찾는 식으로 해서 데이터를 down_sampling 하여 취합함.

##취득 데이터(raw_merge.py)
1) X,Y,Z : 3-AXIS Stage Motor 좌표
- X,Y에는 모터가 다음 좌표로 움직일때의 좌표가 포함되어 있음.
- Z에는 실제로 접촉하지 않았을때의 물리적 모터 좌표가 포함되어 있음.
2) Fz : FT센서로 기록된 값 Force[N]값
3) d : Hemi-Sphere Indentor Diameter


## INPUT Data
1) Input1 = [S1,S2,...,S15,16]
2) Input2 = \delta[S1_base,S2_base,...,S15_base,16_base]_{\theta}
3) '~/sensor_training/raw_data'

## Ground True 생성 FOR SATS
1) GT1: Input1 데이터를 위한 GT값을 생성해야 SATS를 사용할 수 있음.
	1.1) 20260402_GT.md 내용 참고하여 셀에 가해지는 응력분포 값을 GT로 주어질 것임.
2) GT2:  Input2 센서가 밴딩된 상태에서의 값에 대한 밴딩된 각도값(on-hot 구성)

## 학습 모듈1 SATS 구성
0) Input1 , GT1
1) LSTM Modul
2) Self-Attention Modul
3) Local Map Modul
4) CNN Module

## 학습 모듈2 MLP 구성
0) INPUT2, GT2
1) MLP 구성을 통해 Baseline이 변했다는 것을 인지하여 SATS 구성에 연결하여 BASELINE 보상을 넘겨줌.
2) Head를 따로 두어 Bending 된 상태 출력.

## OUTPUT/INFERENCE
1) SATS를 이용한 [40x40] Grid Virtual Sensing Unit 추론하기.
2) 센서가 밴딩된 상태(Bending_DEG)를 추론하고, 이런 상태를 보상/보완하여 1)기능을 확보/보장할 수 있는 추론
