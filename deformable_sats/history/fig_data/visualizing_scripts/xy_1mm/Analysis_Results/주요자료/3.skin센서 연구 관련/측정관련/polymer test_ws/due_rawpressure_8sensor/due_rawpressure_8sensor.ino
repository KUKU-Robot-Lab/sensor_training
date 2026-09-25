#include <SPI.h>
#include "bmp3.h"

#define NUM_SENSORS 8
uint8_t csPins[NUM_SENSORS] = { 13, 12, 11, 10, 9, 8, 7, 6 };
const uint32_t SPI_CLOCK = 10000000;  // 또는 1000000 = 1MHz


bmp3_dev bmpSensors[NUM_SENSORS];
bmp3_uncomp_data uncomp_data_array[NUM_SENSORS];

int8_t spi_read(uint8_t reg_addr, uint8_t *data, uint32_t len, void *intf_ptr);
int8_t spi_write(uint8_t reg_addr, const uint8_t *data, uint32_t len, void *intf_ptr);
void bmp3_delay_wrapper(uint32_t period, void *intf_ptr);
void selectSensor(int index);
void deselectAll();

void setup() {
  Serial.begin(460800);
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
  for (int i = 0; i < NUM_SENSORS; i++) {
    selectSensor(i);

    bmp3_settings settings = { 0 };
    settings.op_mode = BMP3_MODE_FORCED;
    bmp3_set_op_mode(&settings, &bmpSensors[i]);
    //delay(10);

    uint8_t data[6] = { 0 };
    bmp3_get_regs(BMP3_REG_DATA, data, 6, &bmpSensors[i]);

    uncomp_data_array[i].pressure = (uint32_t)data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16);
    deselectAll();
  }

  for (int i = 0; i < NUM_SENSORS; i++) {
    Serial.print((uint32_t)uncomp_data_array[i].pressure);
    if (i < NUM_SENSORS - 1) {
      Serial.print(",");
    } else {
      Serial.println();
    }
  }

  //delay(5);
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
