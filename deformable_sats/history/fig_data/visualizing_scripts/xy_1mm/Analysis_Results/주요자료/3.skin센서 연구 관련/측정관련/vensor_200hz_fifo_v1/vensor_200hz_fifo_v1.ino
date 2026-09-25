
#include <SPI.h>
#include "bmp3.h"

// =================================================================================
// 사용자 설정 (User Settings)
// =================================================================================

// --- 시리얼 통신 설정 ---
#define SERIAL_BAUDRATE 250000 // 시리얼 통신 속도 (Baudrate)

// --- IIR 필터 설정 ---
#define USE_IIR_FILTER 1 // 1: 사용, 0: 미사용
#if USE_IIR_FILTER
  const uint8_t IIR_FILTER_COEFFICIENT = BMP3_IIR_FILTER_COEFF_3;
#endif

// --- FIFO 설정 ---
// FIFO 워터마크 레벨: 센서가 N개의 데이터를 모으면 알려줍니다.
// 10으로 설정 시, 200Hz에서 50ms마다 (5ms * 10 = 50ms) 데이터를 한꺼번에 읽어옵니다.
#define FIFO_WATERMARK_FRAMES 10

// =================================================================================

#define NUM_SENSORS 16
#define PRESSURE_FRAME_SIZE 4 // FIFO 압력 데이터 1프레임 크기 (헤더 1 + 데이터 3)

uint8_t csPins[NUM_SENSORS] = { 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37 };
const uint32_t SPI_CLOCK = 10000000;  // 10MHz

const uint8_t FRAME_HEADER = 0xAA;
const uint8_t FRAME_FOOTER = 0x55;

bmp3_dev bmpSensors[NUM_SENSORS];

// 16개 센서 * 10개 프레임의 압력 데이터를 저장할 버퍼
uint32_t pressure_burst_data[NUM_SENSORS][FIFO_WATERMARK_FRAMES];

// 함수 프로토타입
int8_t spi_read(uint8_t reg_addr, uint8_t *data, uint32_t len, void *intf_ptr);
int8_t spi_write(uint8_t reg_addr, const uint8_t *data, uint32_t len, void *intf_ptr);
void bmp3_delay_wrapper(uint32_t period, void *intf_ptr);
void selectSensor(int index);
void deselectAll();

void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  SPI.begin();

  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(csPins[i], OUTPUT);
    digitalWrite(csPins[i], HIGH);

    bmpSensors[i].intf = BMP3_SPI_INTF;
    bmpSensors[i].read = spi_read;
    bmpSensors[i].write = spi_write;
    bmpSensors[i].intf_ptr = &csPins[i];
    bmpSensors[i].delay_us = bmp3_delay_wrapper;
    bmpSensors[i].dummy_byte = 1;

    selectSensor(i);
    
    if (bmp3_init(&bmpSensors[i]) != BMP3_OK) {
      while(1); // 초기화 실패 시 정지
    }

    // 1. ODR, OSR, IIR 등 기본 설정
    uint32_t settings_selection = BMP3_SEL_PRESS_EN | BMP3_SEL_TEMP_EN | BMP3_SEL_PRESS_OS | BMP3_SEL_ODR;
    bmp3_settings settings = { 0 };
    settings.press_en = BMP3_ENABLE;
    settings.temp_en = BMP3_DISABLE;
    settings.odr_filter.press_os = BMP3_NO_OVERSAMPLING;
    settings.odr_filter.odr = BMP3_ODR_200_HZ;

    #if USE_IIR_FILTER
      settings.odr_filter.iir_filter = IIR_FILTER_COEFFICIENT;
      settings_selection |= BMP3_SEL_IIR_FILTER;
    #endif

    bmp3_set_sensor_settings(settings_selection, &settings, &bmpSensors[i]);

    // 2. FIFO 설정
    bmp3_fifo_settings fifo_settings = {0};
    fifo_settings.mode = BMP3_ENABLE;           // FIFO 활성화
    fifo_settings.press_en = BMP3_ENABLE;       // FIFO에 압력 데이터 저장
    fifo_settings.temp_en = BMP3_DISABLE;       // 온도는 저장 안함
    fifo_settings.time_en = BMP3_DISABLE;       // 센서 시간 저장 안함
    fifo_settings.stop_on_full_en = BMP3_ENABLE; // FIFO 꽉 차면 정지
    
    uint16_t fifo_settings_selection = BMP3_SEL_FIFO_MODE | BMP3_SEL_FIFO_PRESS_EN | BMP3_SEL_FIFO_TEMP_EN | BMP3_SEL_FIFO_TIME_EN | BMP3_SEL_FIFO_STOP_ON_FULL_EN;
    bmp3_set_fifo_settings(fifo_settings_selection, &fifo_settings, &bmpSensors[i]);

    // 3. FIFO 워터마크 설정 (N개 데이터가 쌓이면 인터럽트 발생)
    struct bmp3_fifo_data fifo_data;
    fifo_data.req_frames = FIFO_WATERMARK_FRAMES;
    bmp3_set_fifo_watermark(&fifo_data, &fifo_settings, &bmpSensors[i]);

    // 4. Normal 모드 시작
    settings.op_mode = BMP3_MODE_NORMAL;
    bmp3_set_op_mode(&settings, &bmpSensors[i]);

    deselectAll();
  }
}

void loop() {
  // loop는 주기적으로 FIFO 버퍼를 확인하는 역할만 수행합니다.
  // 실제 데이터 읽기는 약 50ms 마다 (워터마크 설정에 따라) 한 번씩 일어납니다.
  read_and_send_data_burst();
  
  // 루프가 너무 빨리 도는 것을 방지. 
  // 실제 데이터 처리 주기는 아래 딜레이가 아닌 워터마크 레벨에 의해 결정됩니다.
  delay(1);
}

void read_and_send_data_burst() {
  uint16_t fifo_length = 0;
  const uint16_t required_bytes = FIFO_WATERMARK_FRAMES * PRESSURE_FRAME_SIZE;
  
  // 첫 번째 센서의 FIFO 길이만 확인하여 모든 센서의 데이터 수집 시점을 동기화
  selectSensor(0);
  bmp3_get_fifo_length(&fifo_length, &bmpSensors[0]);
  deselectAll();

  if (fifo_length >= required_bytes) {
    // 모든 센서에서 데이터 버스트 읽기
    for (int i = 0; i < NUM_SENSORS; i++) {
      selectSensor(i);
      
      uint8_t fifo_buffer[required_bytes];
      bmp3_get_regs(BMP3_REG_FIFO_DATA, fifo_buffer, required_bytes, &bmpSensors[i]);
      
      // FIFO 버퍼에서 순수 압력 데이터 파싱
      int frame_count = 0;
      for (int j = 0; j < required_bytes; j += PRESSURE_FRAME_SIZE) {
        if (fifo_buffer[j] == BMP3_FIFO_PRESS_FRAME) { // 압력 프레임 헤더 확인
          if (frame_count < FIFO_WATERMARK_FRAMES) {
            pressure_burst_data[i][frame_count] = (uint32_t)fifo_buffer[j+1] | ((uint32_t)fifo_buffer[j+2] << 8) | ((uint32_t)fifo_buffer[j+3] << 16);
            frame_count++;
          }
        }
      }
      deselectAll();
    }

    // PC로 데이터 전송
    Serial.write(FRAME_HEADER);
    Serial.write((uint8_t*)pressure_burst_data, sizeof(pressure_burst_data));
    Serial.write(FRAME_FOOTER);
  }
}


// --- 이하 헬퍼 함수들은 기존과 동일 ---

void selectSensor(int index) {
  deselectAll();
  digitalWrite(csPins[index], LOW);
}

void deselectAll() {
  for (int i = 0; i < NUM_SENSORS; i++) {
    digitalWrite(csPins[i], HIGH);
  }
}

int8_t spi_read(uint8_t reg_addr, uint8_t *data, uint32_t len, void *intf_ptr) {
  uint8_t cs = *(uint8_t *)intf_ptr;
  digitalWrite(cs, LOW);
  SPI.beginTransaction(SPISettings(SPI_CLOCK, MSBFIRST, SPI_MODE0));
  SPI.transfer(reg_addr | 0x80);
  for (uint32_t i = 0; i < len; i++) {
    data[i] = SPI.transfer(0x00);
  }
  SPI.endTransaction();
  digitalWrite(cs, HIGH);
  return BMP3_OK;
}

int8_t spi_write(uint8_t reg_addr, const uint8_t *data, uint32_t len, void *intf_ptr) {
  uint8_t cs = *(uint8_t *)intf_ptr;
  digitalWrite(cs, LOW);
  SPI.beginTransaction(SPISettings(SPI_CLOCK, MSBFIRST, SPI_MODE0));
  SPI.transfer(reg_addr & 0x7F);
  for (uint32_t i = 0; i < len; i++) {
    SPI.transfer(data[i]);
  }
  SPI.endTransaction();
  digitalWrite(cs, HIGH);
  return BMP3_OK;
}

void bmp3_delay_wrapper(uint32_t period, void *intf_ptr) {
  delayMicroseconds(period);
}
