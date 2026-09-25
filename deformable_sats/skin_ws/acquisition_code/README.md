# Tactile Acquisition Code

이 폴더의 최종 목적은 촉각 센서 정보와 모터 정보를 한 번에 취득하는 것이다. 최종 로거는 루트의 `final_logger.py`이며, 나머지 파일은 보조 도구 또는 과거 코드로 폴더 안에 정리되어 있다.

## 가장 중요한 파일

루트에 남겨둔 파일만 먼저 보면 된다.

```text
final_logger.py                 최종 통합 raw binary logger
loadcell_bin_logger.py          final_logger.py가 사용하는 로드셀 raw helper
serial_autodetect.py            DUE/로드셀 COM 포트 자동 인식 helper
convert_bins.py                 raw binary -> CSV 변환
README.md                       이 문서
README_LOADCELL_INDICATOR.md    CI-400AL 로드셀 인디케이터 설정
```

`final_logger.py`는 `loadcell_bin_logger.py`와 `serial_autodetect.py`를 import하므로 세 파일은 같은 폴더에 둔다.

## 현재 장비 설정

```text
DUE serial:       COM11, 250000 bps (자동 인식, 아래 참고)
Loadcell serial:  COM10, 115200 bps, 8N1 (자동 인식, 아래 참고)
AFD50 CAN:        ixxat, channel 0, 1 Mbps, force id 0x01A
EtherMotion DLL:  C:\Program Files (x86)\PAIX\NMC\DLL\x64\NMC2.dll
EtherMotion dev:  11
EtherMotion group: 0
```

### DUE / 로드셀 포트 자동 인식

USB 포트가 바뀌어도 코드를 수정할 필요가 없도록 `final_logger.py`와 `loadcell_bin_logger.py`는 실행 시 연결된 장치를 스캔해서 포트를 자동으로 찾는다.

- DUE: VID:PID `2341:003D`/`2341:003E` 또는 설명에 `arduino`/`due`가 포함된 포트
- Loadcell: VID:PID `067B:2303`(Prolific USB-RS232 어댑터) 또는 설명에 `prolific`/`usb-to-serial`/`usb serial`이 포함된 포트

우선순위는 `--due-port`/`--loadcell-port` CLI 인자 > `DUE_PORT`/`LOADCELL_PORT` 환경변수 > 자동 인식 결과(정확히 1개 매칭될 때) > 코드 상단의 하드코딩된 기본값(`COM11`/`COM10`) 순이다. 후보가 2개 이상이거나 하나도 없으면 콘솔에 상황을 출력하고 하드코딩된 기본값으로 폴백한다.

```powershell
# 포트를 강제로 지정하고 싶을 때
python final_logger.py --due-port COM7 --loadcell-port COM9

# 또는 환경변수로
$env:DUE_PORT = "COM7"; python final_logger.py
```

`tools\device_tests\due_reader.py`, `due_writer.py`, `loadcell_logger.py` (단독 장치 테스트용)도 동일한 자동 인식을 사용한다. `due_reader.py`/`due_writer.py`는 연결이 끊겼다 재시도할 때마다 다시 스캔하므로 케이블을 다른 USB 포트에 꽂아도 재연결된다. `loadcell_logger.py`는 `loadcell_bin_logger.py`와 동일하게 `--port`/`LOADCELL_PORT` 환경변수로 override 가능하다.

AFD50(CAN 채널)과 EtherMotion(NMC DLL dev_no)은 COM 포트를 쓰지 않아 이 자동 인식 대상이 아니다 — 각각 `CAN_CHANNEL`/`--dev-no`로 직접 지정한다.

로드셀 인디케이터 설정은 [README_LOADCELL_INDICATOR.md](README_LOADCELL_INDICATOR.md)를 참고한다.

## 기본 실행 위치

명령은 여기에서 실행한다.

```powershell
cd C:\Users\SORO7\Desktop\Tactile\skin_ws\acquisition_code
```

확인된 Python 경로:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe
```

## 최종 로깅

통합 raw binary 로깅:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe final_logger.py
```

기본 동작:

- DUE, AFD50, loadcell reader를 시작한다.
- EtherMotion DLL이 열리면 EtherMotion 위치/명령 정보도 기록한다.
- 로깅 시작 전 약 2초 동안 로드셀 baseline을 자동 측정한다.
- 기본 시작 트리거는 `ethermotion`이다.
- 저장 위치는 현재 폴더 안의 `log`이다.
- 실행마다 `YYYYMMDD_testN` 세션 폴더를 새로 만든다.

저장 예:

```text
C:\Users\SORO7\Desktop\Tactile\skin_ws\acquisition_code\log\20260717_test1\
  due_raw_burst_YYYYMMDD_HHMMSS.bin
  afd50_can_raw_YYYYMMDD_HHMMSS.bin
  loadcell_raw_YYYYMMDD_HHMMSS.bin
  ethermotion_encoder_YYYYMMDD_HHMMSS.bin
```

실행 중 아래 값이 증가하면 정상이다.

```text
DUE=... burst/s
AFD50=... msg/s
Loadcell=... chunk/s, ... B/s
EtherMotion=... rec/s
LC=...B
```

로드셀 baseline이 잡히면 시작 전에 아래와 비슷한 메시지가 나온다.

```text
Loadcell baseline: mean=..., std=..., p2p=..., samples=...
```

## 시작 옵션

EtherMotion 동작 감지 후 시작:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe final_logger.py --start-trigger ethermotion
```

즉시 시작:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe final_logger.py --start-trigger immediate
```

수동 시작:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe final_logger.py --start-trigger manual
```

EtherMotion 폴링 속도 지정:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe final_logger.py --ethermotion-hz 1000
```

현재 코드 기본값은 `--ethermotion-hz 0`이며, 0은 sleep 없이 가능한 빠르게 폴링하는 설정이다.

## CSV 변환

`final_logger.py`는 CSV를 직접 쓰지 않고 raw binary를 저장한다. CSV가 필요하면 로깅 후 변환한다.

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe convert_bins.py
```

기본 동작:

- `acquisition_code\log`에서 아직 CSV가 없는 최신 `YYYYMMDD_testN` 폴더를 찾는다.
- 지원되는 `.bin` 파일을 CSV로 변환한다.
- CSV는 같은 세션 폴더 안에 저장된다.

출력 CSV:

```text
due_data.csv
afd50_data.csv
ethermotion_data.csv
loadcell_data.csv
```

`loadcell_data.csv`는 baseline을 적용한 컬럼을 같이 만든다.

```text
elapsed_ns,time_s,kg_raw,kg_zeroed,baseline_mean_kg
```

- `kg_raw`: 인디케이터에서 받은 원래 kg 값
- `kg_zeroed`: 실행 시작 전 baseline 평균을 뺀 값
- `baseline_mean_kg`: 해당 실행에서 측정된 baseline 평균

AFD50 bias sample 수 지정:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe convert_bins.py --bias-samples 200
```

AFD50 Fz 부호 반전을 끄기:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe convert_bins.py --no-invert-fz
```

## 정리된 폴더 구조

```text
acquisition_code\
  final_logger.py
  loadcell_bin_logger.py
  convert_bins.py
  README.md
  README_LOADCELL_INDICATOR.md
  log\
  tools\
    device_tests\
    gui\
    node_generation\
    analysis\
  legacy\
  references\
    ethermotion_projects\
  archived_data\
    device_outputs\
    temp_tests\
```

각 폴더 역할:

```text
tools\device_tests     DUE/AFD50/EtherMotion/loadcell 단독 테스트 스크립트
tools\gui              GUI logger와 GUI 포맷 변환기
tools\node_generation  EtherMotion node/profile 생성 스크립트
tools\analysis         실험 데이터 확인/분석 보조 스크립트
legacy                 예전 통합 logger 코드
references             EtherMotion project 등 참고 파일
archived_data          예전 출력 데이터와 임시 테스트 파일
```

## 개별 장치 테스트

로드셀 raw 바이트 기록은 루트의 helper를 사용한다.

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe loadcell_bin_logger.py --duration-sec 5 --output test_lc_raw.bin
```

로드셀 CSV 단독 테스트:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe tools\device_tests\loadcell_logger.py --duration-sec 3 --output test_lc.csv
```

포트 목록 확인:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe -c "import serial.tools.list_ports as p; print('\n'.join(f'{x.device}\t{x.description}\t{x.hwid}' for x in p.comports()))"
```

DUE 단독 테스트:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe tools\device_tests\due_reader.py
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe tools\device_tests\due_writer.py
```

AFD50 단독 테스트:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe tools\device_tests\afd50_reader.py
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe tools\device_tests\afd50_writer.py
```

EtherMotion 단독 테스트:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe tools\device_tests\ethermotion_reader.py
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe tools\device_tests\ethermotion_writer.py
```

## 점검 명령

문법 확인:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe -m py_compile final_logger.py convert_bins.py loadcell_bin_logger.py
```

`final_logger.py`가 로드셀 helper를 정상 import하는지 확인:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe -c "import final_logger as f; print(f.LOADCELL_PORT, f.LOADCELL_BAUD, f.open_loadcell_serial is not None)"
```

로드셀 원시 수신 확인:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe -c "import time, serial; s=serial.Serial('COM10',115200,bytesize=8,parity='N',stopbits=1,timeout=0.2); s.reset_input_buffer(); t=time.time()+5; data=b''
while time.time()<t:
    data += s.read(4096)
s.close(); print('bytes', len(data)); print(repr(data[:200]))"
```

정상이라면 `bytes`가 0보다 크고 `ST,GS,... kg` 형식 문자열이 보인다.

## 주의사항

- 같은 COM 포트는 한 번에 하나의 프로그램만 열 수 있다.
- `final_logger.py` 실행 중에 로드셀 단독 로거 또는 시리얼 터미널을 동시에 열면 `PermissionError: access denied`가 날 수 있다.
- `final_logger.py`는 raw binary만 저장한다.
- CSV는 `convert_bins.py`로 후처리한다.
- `convert_bins.py`는 CSV가 아직 없는 최신 세션 폴더만 자동 변환한다.
- 장치 포트가 바뀌면 코드 상단의 포트 상수 또는 각 스크립트의 `--port` 옵션을 확인한다.
