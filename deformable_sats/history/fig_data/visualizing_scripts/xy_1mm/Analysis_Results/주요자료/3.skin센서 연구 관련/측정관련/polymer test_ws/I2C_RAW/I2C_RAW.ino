#include <Wire.h>
#include "bmp3.h"

#define BMP3_ADDR_I2C 0x77  // I2C 주소 확인 필요

// BMP3 객체
struct bmp3_dev bmp3Device;
struct bmp3_uncomp_data rawData;
struct bmp3_settings settings = {0}; // 구조체를 0으로 초기화

// I2C read 함수
int8_t bmp3_i2c_read(uint8_t reg_addr, uint8_t *data, uint32_t len, void *intf_ptr) {
  Wire.beginTransmission(*(uint8_t*)intf_ptr);
  Wire.write(reg_addr);
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom(*(uint8_t*)intf_ptr, (uint8_t)len) != len) {
    return -1; // 요청한 만큼의 데이터를 받지 못함
  }
  Wire.readBytes(data, len);
  return BMP3_OK;
}

// I2C write 함수
int8_t bmp3_i2c_write(uint8_t reg_addr, const uint8_t *data, uint32_t len, void *intf_ptr) {
  Wire.beginTransmission(*(uint8_t*)intf_ptr);
  Wire.write(reg_addr);
  Wire.write(data, len);
  return (Wire.endTransmission() == 0) ? BMP3_OK : -1;
}

// Delay 함수
void bmp3_delay_us(uint32_t period, void*) {
  delayMicroseconds(period);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  static uint8_t i2c_addr = BMP3_ADDR_I2C;
  bmp3Device.intf = BMP3_I2C_INTF;
  bmp3Device.intf_ptr = &i2c_addr;
  bmp3Device.read = bmp3_i2c_read;
  bmp3Device.write = bmp3_i2c_write;
  bmp3Device.delay_us = bmp3_delay_us;

  if (bmp3_init(&bmp3Device) != BMP3_OK) {
    Serial.println("Sensor init failed.");
    while (1);
  }

  settings.press_en = BMP3_ENABLE;
  settings.temp_en = BMP3_DISABLE;
  settings.odr_filter.press_os = BMP3_NO_OVERSAMPLING;
  settings.op_mode = BMP3_MODE_FORCED;

  uint32_t settings_sel = BMP3_SEL_PRESS_EN | BMP3_SEL_PRESS_OS | BMP3_SEL_TEMP_EN;

  if (bmp3_set_sensor_settings(settings_sel, &settings, &bmp3Device) != BMP3_OK) {
    Serial.println("Failed to set sensor settings.");
    while (1);
  }
}

void loop() {
  // Forced mode 설정
  if (bmp3_set_op_mode(&settings, &bmp3Device) != BMP3_OK) {
    Serial.println("Failed to set op mode.");
    return;
  }

  bmp3Device.delay_us(40000, nullptr); // 측정 시간 대기

  // 압력 데이터는 3바이트이므로 3바이트만 읽습니다.
  uint8_t reg_data[3] = {0};
  if (bmp3_get_regs(BMP3_REG_DATA, reg_data, 3, &bmp3Device) != BMP3_OK) {
    // Python에서 파싱 오류를 일으키지 않도록 명확한 에러 메시지를 보냅니다.
    Serial.println("Error: Read failed");
    return;
  }

  // raw pressure 추출 (24비트)
  rawData.pressure = (uint32_t)reg_data[2] << 16 | (uint32_t)reg_data[1] << 8 | reg_data[0];
  Serial.println((unsigned long)rawData.pressure);  // Raw ADC count 출력

  delay(20); // Python 스크립트의 interval과 비슷하게 조절
}