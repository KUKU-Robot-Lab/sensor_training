# CI-400AL Loadcell Indicator Setup

이 문서는 로드셀 -> CI-400AL 인디케이터 -> PC 연결에서 RS-232C 스트림 데이터를 받기 위한 설정과 점검 절차를 정리한 것이다.

## 현재 확인된 상태

```text
PC port: COM10
Device: Prolific USB-to-Serial Comm Port
Serial: 115200 bps, 8 data bits, no parity, 1 stop bit
Observed stream: ST,GS,...,   0.000 kg\r\n
```

확인 결과 `COM10`에서 5초 동안 약 `22478 bytes`가 수신되었고, `loadcell_logger.py` 3초 테스트에서 약 `603 rows`가 CSV로 기록되었다.

## COM1 사용 시 인디케이터 설정

```text
F1.01 = 6   AD 320 Hz
F2.02 = 0   실시간 전송
F2.03 = 0   COM1: 8 data bits, 1 stop bit, parity none
F2.04 = 7   COM1: 115200 bps
F2.05 = 0   COM1: 표시값 송신
F2.06 = 0   COM1: CAS 22-byte format
F2.07 = 2   COM1: Stream Mode, 안정/불안정 모두 전송
```

## COM2 사용 시 인디케이터 설정

```text
F2.08 = 0   COM2: 8 data bits, 1 stop bit, parity none
F2.09 = 7   COM2: 115200 bps
F2.10 = 0   COM2: 표시값 송신
F2.11 = 0   COM2: CAS 22-byte format
F2.12 = 2   COM2: Stream Mode, 안정/불안정 모두 전송
```

실제 케이블이 꽂힌 인디케이터 단자가 COM1인지 COM2인지 확인해야 한다. COM1 단자에 꽂았으면 `F2.03`~`F2.07`, COM2 단자에 꽂았으면 `F2.08`~`F2.12`가 적용된다.

## RS-232C 배선

매뉴얼 기준 PC 연결은 아래와 같다.

```text
Indicator TXD -> PC RXD
Indicator RXD -> PC TXD
Indicator GND -> PC GND
```

DB9 PC/USB-RS232 기준으로는 보통 아래 핀을 사용한다.

```text
Indicator TXD -> PC pin 2
Indicator RXD -> PC pin 3
Indicator GND -> PC pin 5
```

설정이 맞는데도 수신 바이트가 `0`이면 TXD/RXD 크로스 여부, GND 연결, COM1/COM2 단자 불일치를 먼저 확인한다.

## Raw 기록

인디케이터가 보내는 바이트를 최대한 그대로 저장하려면 `loadcell_bin_logger.py`를 사용한다.

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe loadcell_bin_logger.py
```

짧은 테스트:

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe loadcell_bin_logger.py --duration-sec 5 --output test_lc_raw.bin
```

정상이라면 종료 시 아래 값이 0보다 커야 한다.

```text
Chunks written: ...
Payload bytes written: ...
```

## CSV 기록

kg 숫자만 뽑아서 CSV로 저장하려면 `loadcell_logger.py`를 사용한다.

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe loadcell_logger.py --duration-sec 3 --output test_lc.csv
```

정상이라면 `Rows written` 값이 증가한다.

## 원시 수신 테스트

```powershell
C:\Users\SORO7\AppData\Local\Programs\Python\Python313-32\python.exe -c "import time, serial; s=serial.Serial('COM10',115200,bytesize=8,parity='N',stopbits=1,timeout=0.2); s.reset_input_buffer(); t=time.time()+5; data=b''
while time.time()<t:
    data += s.read(4096)
s.close(); print('bytes', len(data)); print(repr(data[:500]))"
```

정상 예:

```text
bytes 22478
b'ST,GS,...,   0.000 kg\r\n...'
```

비정상 예:

```text
bytes 0
b''
```

`bytes 0`이면 Python 코드보다 인디케이터 출력 설정, RS-232 배선, COM1/COM2 설정, USB-RS232 연결 문제를 우선 확인한다.

## 주의사항

- 같은 COM 포트는 한 번에 하나의 프로그램만 열 수 있다.
- `final_logger.py`, `loadcell_logger.py`, `loadcell_bin_logger.py`, 시리얼 터미널을 동시에 열면 하나는 `PermissionError: access denied`가 날 수 있다.
- raw 보존 목적이면 `loadcell_bin_logger.py`를 우선 사용한다.
- CSV는 kg 숫자만 남기므로 상태 플래그와 원본 바이트는 사라진다.
- `F1.01 = 6`으로 AD 320 Hz 설정이어도 실제 RS-232 출력 행 수는 포맷과 인디케이터 처리 주기에 따라 약 200 Hz 정도로 관측될 수 있다.

