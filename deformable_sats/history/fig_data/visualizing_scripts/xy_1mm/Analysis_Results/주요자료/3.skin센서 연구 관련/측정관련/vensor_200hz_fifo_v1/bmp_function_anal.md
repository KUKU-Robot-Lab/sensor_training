
# BMP384 주요 함수 분석 (BMP3 C 드라이버 기준)

이 문서는 `vensor_200hz_fifo_v1.ino` 예제에서 사용된 Bosch Sensortec의 BMP3 드라이버 주요 함수들의 기능과 설정을 데이터시트와 연관지어 설명합니다.

---

### 1. `bmp3_init()`

- **기능**: 센서를 초기화합니다.
- **설명**:
  - 센서와의 통신(SPI/I2C)을 준비하고 칩 ID를 읽어 센서가 정상적으로 연결되었는지 확인합니다.
  - 이후 센서 내부에 저장된 고유의 보정값(Calibration Data)을 읽어와 `bmp3_dev` 구조체에 저장합니다. 이 보정값은 나중에 원시 데이터를 실제 압력(Pa)으로 변환하는 데 사용됩니다.
  - 모든 센서 관련 작업을 시작하기 전에 반드시 호출해야 하는 가장 기본적인 함수입니다.
  - (데이터시트 3. Functional description, p.8)
- **간단한 설정**:
  ```c
  struct bmp3_dev my_sensor;
  // ... my_sensor 구조체의 read, write, delay_us 등 함수 포인터 설정 ...
  int8_t rslt = bmp3_init(&my_sensor);
  if (rslt != BMP3_OK) {
    // 초기화 실패
  }
  ```

---

### 2. `bmp3_set_op_mode()`

- **기능**: 센서의 동작 모드(Power Mode)를 설정합니다.
- **설명**:
  - 센서가 어떤 방식으로 측정할지를 결정합니다.
  - **`BMP3_MODE_SLEEP`**: 기본 상태. 아무 측정 없이 최소 전력으로 대기합니다.
  - **`BMP3_MODE_FORCED`**: 한 번만 압력/온도를 측정하고 `SLEEP` 모드로 돌아갑니다. 저속 측정이 필요할 때 유용합니다.
  - **`BMP3_MODE_NORMAL`**: 설정된 ODR(출력 데이터 속도)에 맞춰 자동으로 계속해서 측정합니다. 저희 프로젝트처럼 연속적인 데이터가 필요할 때 사용합니다.
  - (데이터시트 3.3 Power modes, p.10)
- **간단한 설정**:
  ```c
  bmp3_settings settings = {0};
  settings.op_mode = BMP3_MODE_NORMAL;
  bmp3_set_op_mode(&settings, &my_sensor);
  ```

---

### 3. `bmp3_set_sensor_settings()`

- **기능**: 센서의 세부 측정 파라미터를 설정합니다.
- **설명**:
  - 압력/온도 측정 활성화, 오버샘플링(Oversampling), 출력 데이터 속도(ODR), IIR 필터 등 핵심 성능에 관련된 항목들을 설정합니다.
  - 비트마스크(Bitmask)를 통해 변경하려는 항목을 명시적으로 선택(`BMP3_SEL_ODR` 등)하고, `bmp3_settings` 구조체에 원하는 값을 담아 전달합니다.
  - **ODR (Output Data Rate)**: 얼마나 자주 측정할지를 결정합니다. `BMP3_ODR_200_HZ`로 설정 시 5ms 주기로 측정합니다. (데이터시트 4.3.19, p.37)
  - **Oversampling (OSR)**: 측정 노이즈를 줄이기 위해 내부적으로 여러 번 측정하여 평균내는 기능입니다. 높일수록 정밀해지지만 측정 시간이 길어집니다. (데이터시트 3.4.1, p.13)
  - **IIR Filter**: 급격한 노이즈를 줄여주는 디지털 필터입니다. 로봇의 진동이나 외부 충격에 의한 노이즈를 줄이는 데 효과적입니다. (데이터시트 3.4.3, p.14)
- **간단한 설정**:
  ```c
  uint32_t selection = BMP3_SEL_ODR | BMP3_SEL_IIR_FILTER;
  bmp3_settings settings = {0};
  settings.odr_filter.odr = BMP3_ODR_200_HZ; // 200Hz
  settings.odr_filter.iir_filter = BMP3_IIR_FILTER_COEFF_3; // IIR 필터 계수 3
  bmp3_set_sensor_settings(selection, &settings, &my_sensor);
  ```

---

### 4. `bmp3_set_fifo_settings()`

- **기능**: FIFO(First-In, First-Out) 버퍼의 동작 방식을 설정합니다.
- **설명**:
  - 512바이트 크기의 내장 메모리(FIFO)를 어떻게 사용할지 결정합니다.
  - `mode`: FIFO 기능 자체를 켜거나 끕니다.
  - `press_en`, `temp_en`, `time_en`: 압력, 온도, 센서 시간 데이터 중 무엇을 FIFO에 저장할지 선택합니다.
  - `stop_on_full_en`: FIFO가 꽉 찼을 때, 새로운 데이터로 덮어쓸지(Streaming mode), 아니면 저장을 멈출지(Stop-on-full mode) 결정합니다.
  - (데이터시트 3.6 FIFO Description, p.17)
- **간단한 설정**:
  ```c
  bmp3_fifo_settings fifo_settings = {0};
  fifo_settings.mode = BMP3_ENABLE;       // FIFO 활성화
  fifo_settings.press_en = BMP3_ENABLE;   // 압력 데이터 저장
  uint16_t selection = BMP3_SEL_FIFO_MODE | BMP3_SEL_FIFO_PRESS_EN;
  bmp3_set_fifo_settings(selection, &fifo_settings, &my_sensor);
  ```

---

### 5. `bmp3_set_fifo_watermark()`

- **기능**: FIFO 워터마크 레벨(데이터 개수)을 설정합니다.
- **설명**:
  - FIFO 버퍼에 데이터가 몇 개(프레임 단위) 쌓였을 때 사용자에게 알려줄지(인터럽트 발생) 그 기준점을 설정합니다.
  - 예를 들어 10으로 설정하면, 10개의 데이터 프레임이 쌓일 때마다 특정 상태 플래그가 활성화됩니다.
  - 프로세서가 매번 센서를 확인하는 대신, 데이터가 일정량 모였을 때만 처리하게 하여 효율을 높입니다.
  - (데이터시트 3.7.5.1 FIFO watermark interrupt, p.22)
- **간단한 설정**:
  ```c
  struct bmp3_fifo_data fifo_data;
  fifo_data.req_frames = 10; // 워터마크를 10 프레임으로 설정
  bmp3_set_fifo_watermark(&fifo_data, &fifo_settings, &my_sensor);
  ```

---

### 6. `bmp3_get_fifo_length()`

- **기능**: FIFO 버퍼에 현재 저장된 데이터의 총 크기(바이트 단위)를 가져옵니다.
- **설명**:
  - FIFO에서 데이터를 읽기 전에, 얼마나 많은 데이터가 수신 대기 중인지 확인할 때 사용합니다.
  - 이 함수가 반환하는 값과 워터마크 바이트 크기를 비교하여 데이터를 읽을 시점을 결정할 수 있습니다.
  - (데이터시트 4.3.9 FIFO_LENGTH, p.32)
- **간단한 설정**:
  ```c
  uint16_t bytes_in_fifo;
  bmp3_get_fifo_length(&bytes_in_fifo, &my_sensor);
  ```

---

### 7. `bmp3_get_regs()`

- **기능**: 특정 레지스터 주소에서 원하는 길이만큼의 원시(raw) 데이터를 직접 읽습니다.
- **설명**:
  - 드라이버의 다른 고수준 함수들을 거치지 않고, 하드웨어 레지스터에 직접 접근하는 저수준(low-level) 함수입니다.
  - 저희 프로젝트에서는 이 함수를 사용하여 FIFO 데이터 레지스터(`BMP3_REG_FIFO_DATA`)에서 쌓여있는 전체 데이터를 한 번의 버스트 읽기(burst read)로 빠르게 가져옵니다.
  - (데이터시트 4.3 Register description, p.30)
- **간단한 설정**:
  ```c
  uint8_t fifo_buffer[40];
  // FIFO 데이터 레지스터에서 40바이트를 읽어옴
  bmp3_get_regs(BMP3_REG_FIFO_DATA, fifo_buffer, 40, &my_sensor);
  ```

---

## 알아두면 좋은 추가 함수

### 8. `bmp3_get_sensor_data()`

- **기능**: **보정된** 압력 및 온도 데이터를 가져옵니다.
- **설명**:
  - 센서에서 원시 데이터를 읽고, `bmp3_init`에서 가져온 보정 계수를 사용하여 실제 압력(Pa)과 온도(°C) 값으로 변환해주는 고수준 함수입니다.
  - **주의**: 이 함수는 내부적으로 부동소수점 연산을 수행하므로, 16개 센서에 대해 200Hz로 호출하면 Arduino Due에 상당한 연산 부하를 줄 수 있습니다. 저희 프로젝트처럼 최고 속도가 필요할 때는 원시 데이터를 PC로 보내 보정하는 것이 유리할 수 있습니다.
  - (데이터시트 9. Appendix, p.55-56)
- **간단한 설정**:
  ```c
  struct bmp3_data compensated_data;
  // 압력 데이터만 요청
  bmp3_get_sensor_data(BMP3_PRESS, &compensated_data, &my_sensor);
  // compensated_data.pressure 에 보정된 압력(Pa) 값이 저장됨
  // (단, 정수형 보상 모드일 경우 100이 곱해진 값이 저장됨)
  ```

---

### 9. `bmp3_soft_reset()`

- **기능**: 센서를 소프트웨어적으로 리셋합니다.
- **설명**:
  - 전원을 껐다 켜는 것과 유사하게, 센서의 모든 레지스터를 기본값으로 초기화합니다.
  - 센서가 예상치 못하게 멈추거나 이상하게 동작할 때, 프로그램을 재시작하지 않고 센서만 초기 상태로 되돌리고 싶을 때 유용합니다.
  - (데이터시트 4.3.22 CMD, p.38)
- **간단한 설정**:
  ```c
  bmp3_soft_reset(&my_sensor);
  // 리셋 후에는 일반적으로 센서 설정(init, set_settings 등)을 다시 해주어야 합니다.
  ```

---

### 10. `bmp3_get_status()`

- **기능**: 센서의 현재 상태(데이터 준비 완료, 에러 등)를 확인합니다.
- **설명**:
  - `struct bmp3_status` 구조체를 통해 다양한 상태 정보를 얻을 수 있습니다.
  - `status.sensor.drdy_press`: 압력 측정이 완료되어 읽을 준비가 되었는지 확인.
  - `status.intr.fifo_wm`: FIFO 워터마크 인터럽트 발생 여부 확인.
  - `status.err.conf`: 센서 설정에 오류가 없는지 확인.
  - Polling 방식에서 데이터가 준비되었는지 정확히 확인할 때 유용합니다.
  - (데이터시트 4.3.3 STATUS, p.31)
- **간단한 설정**:
  ```c
  struct bmp3_status status;
  bmp3_get_status(&status, &my_sensor);
  if (status.sensor.drdy_press) {
    // 압력 데이터를 읽을 준비 완료
  }
  ```

---

### 11. `bmp3_fifo_flush()`

- **기능**: FIFO 버퍼의 모든 데이터를 즉시 삭제합니다.
- **설명**:
  - FIFO 버퍼를 깨끗하게 비우고 싶을 때 사용합니다.
  - 예를 들어, 특정 작업(딸기 잡기 등)을 시작하기 직전에 이전에 쌓인 데이터를 모두 버리고, 그 시점부터의 새로운 데이터만 받고 싶을 때 유용합니다.
  - (데이터시트 3.6.7 FIFO flush conditions, p.20)
- **간단한 설정**:
  ```c
  bmp3_fifo_flush(&my_sensor);
  ```

---

# 심화: 커스텀 촉각 센서 교정 방법

폴리머를 부착한 BMP384 센서를 실제 힘(Force) 센서로 사용하기 위한 교정은 2단계로 나누어 생각하는 것이 가장 이상적입니다.

**`[센서 Raw ADC 값]`** → **`1단계: 센서 자체 보정`** → **`[보정된 압력(Pa)]`** → **`2단계: 기구/재료 교정`** → **`[실제 힘(N)]`**

### 1단계: 센서 자체 보정 (Bosch의 역할)

- **목표**: 센서 칩의 제조 오차 및 온도 변화에 따른 특성 변화를 보상하여, 온도에 상관없이 일관된 **압력(Pa)** 값을 얻는 것.
- **방법**: 데이터시트 Appendix 9 (p.55-56)에 명시된 보정식을 사용합니다. 이 식은 센서의 원시 ADC 값(`uncomp_press`, `uncomp_temp`)과 `bmp3_init` 시 읽어온 고유 보정 계수(`PAR_P1`, `PAR_T2` 등)를 사용합니다.
- **구현**: 이 연산은 부동소수점 계산이 많아 Arduino에서 직접 수행하기보다, Arduino에서는 **원시 데이터**를 PC로 전송하고 **PC의 교정 프로그램에서 이 보정식을 적용**하는 것을 강력히 권장합니다. `bmp3.c`의 `compensate_pressure` 함수 로직을 Python 등으로 포팅하여 사용할 수 있습니다.

### 2단계: 기구/재료 교정 (사용자의 역할)

- **목표**: 1단계를 거쳐 얻은 **보정된 압력(Pa)** 과, 기준 장비(상용 F/T 센서)로 측정한 **실제 힘(N)** 사이의 관계를 모델링하는 것. 이 모델은 폴리머의 기계적 특성을 나타냅니다.
- **방법**: 다양한 힘을 가하면서, 그에 해당하는 `보정된 압력(Pa)`과 `실제 힘(N)` 데이터 쌍을 많이 수집합니다. 이 데이터셋을 이용해 둘 사이의 관계를 나타내는 함수(모델)를 찾습니다.
  - **선형 회귀**: `힘(N) = a * 압력(Pa) + b`
  - **다항 회귀**: `힘(N) = a*P² + b*P + c` (비선형 관계를 더 잘 표현)
  - **머신러닝/딥러닝**: 더 복잡한 비선형성이나 히스테리시스(이력 현상)까지 모델링해야 할 경우, 신경망(Neural Network) 등의 모델을 학습시킬 수 있습니다.

### 권장 교정 절차 요약

1.  **데이터 수집**: 로봇 핸드로 물체를 누르며 다양한 힘을 가합니다. 이때 BMP384에서는 **원시 압력/온도 데이터**를, 상용 F/T 센서에서는 **실제 힘(N)** 데이터를 동기화하여 기록합니다.
2.  **PC에서 1단계 보정**: 수집된 모든 BMP384 원시 데이터에 대해 데이터시트의 보정식을 적용하여 **보정된 압력(Pa) 데이터셋**을 만듭니다.
3.  **PC에서 2단계 모델 학습**: `보정된 압력(Pa)`을 입력(X)으로, `실제 힘(N)`을 출력(Y)으로 하여 둘의 관계를 가장 잘 설명하는 회귀 모델을 학습시킵니다.
4.  **실사용 적용**: 실제 로봇이 동작할 때는, BMP384에서 들어오는 원시 데이터를 PC에서 1단계 보정 후, 학습된 2단계 모델에 통과시켜 최종적인 **힘(N)** 값을 실시간으로 추정합니다.

이 2단계 접근법은 '센서 자체의 문제'와 '내가 만든 시스템의 문제'를 분리하여 해결하므로, 훨씬 더 체계적이고 정확한 교정 결과를 얻을 수 있습니다.
