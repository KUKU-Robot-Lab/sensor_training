#include <SPI.h>
#include "bmp3.h"
#define NUM_SENSORS 16
#define FIFO_FRAMES 10
#define FRAME_HEADER 0xAA
#define FRAME_FOOTER 0x55
uint8_t csPins[NUM_SENSORS] = { 25, 24, 23, 22, 29, 28, 27, 26, 33, 32, 31, 30, 37, 36, 35, 34 };
const uint32_t SPI_CLOCK = 10000000;  // 10MHz


bmp3_dev bmpSensors[NUM_SENSORS];
uint32_t pressure_burst_data[NUM_SENSORS][FIFO_FRAMES];

int8_t spi_read(uint8_t reg_addr, uint8_t *data, uint32_t len, void *intf_ptr);
int8_t spi_write(uint8_t reg_addr, const uint8_t *data, uint32_t len, void *intf_ptr);
void bmp3_delay_wrapper(uint32_t period, void *intf_ptr);
void selectSensor(int index);
void deselectAll();
uint32_t readPressureRaw(int sensorIndex);
void writeU32LE(uint32_t value);
void writeBurstPacket();

void setup() {
  Serial.begin(250000);
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
    bmp3_init(&bmpSensors[i]);

    bmp3_settings settings = { 0 };
    settings.press_en = BMP3_ENABLE;
    settings.temp_en = BMP3_DISABLE;
    settings.odr_filter.press_os = BMP3_NO_OVERSAMPLING;
    bmp3_set_sensor_settings(BMP3_SEL_PRESS_EN | BMP3_SEL_PRESS_OS, &settings, &bmpSensors[i]);
    deselectAll();
  }
}

void loop() {
  // 센서 스펙(200Hz)을 활용하기 위해 10프레임을 묶어 바이너리 burst로 전송한다.
  // 패킷 포맷: 0xAA + [16][10] uint32 LE + 0x55
  // payload 순서: sensor-major, frame-minor
  // idx = sensor_i * FIFO_FRAMES + frame_j
  for (int frame_j = 0; frame_j < FIFO_FRAMES; frame_j++) {
    for (int sensor_i = 0; sensor_i < NUM_SENSORS; sensor_i++) {
      pressure_burst_data[sensor_i][frame_j] = readPressureRaw(sensor_i);
    }
  }

  writeBurstPacket();
}

uint32_t readPressureRaw(int sensorIndex) {
  bmp3_settings settings = { 0 };
  settings.op_mode = BMP3_MODE_FORCED;

  selectSensor(sensorIndex);
  bmp3_set_op_mode(&settings, &bmpSensors[sensorIndex]);

  uint8_t data[6] = { 0 };
  bmp3_get_regs(BMP3_REG_DATA, data, 6, &bmpSensors[sensorIndex]);

  uint32_t pressure = (uint32_t)data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16);
  deselectAll();
  return pressure;
}

void writeU32LE(uint32_t value) {
  Serial.write((uint8_t)(value & 0xFF));
  Serial.write((uint8_t)((value >> 8) & 0xFF));
  Serial.write((uint8_t)((value >> 16) & 0xFF));
  Serial.write((uint8_t)((value >> 24) & 0xFF));
}

void writeBurstPacket() {
  Serial.write(FRAME_HEADER);

  for (int sensor_i = 0; sensor_i < NUM_SENSORS; sensor_i++) {
    for (int frame_j = 0; frame_j < FIFO_FRAMES; frame_j++) {
      writeU32LE(pressure_burst_data[sensor_i][frame_j]);
    }
  }

  Serial.write(FRAME_FOOTER);
}

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
