/*
  Laser Sensor Reader Sketch (v3)
  - Calibrates the sensor on startup.
  - Sends a "Laser Ready" signal after calibration is complete.
  - Continuously streams the calculated displacement in mm.
*/

// --- Sensor & Calibration Definitions ---
const int SENSOR_PIN = A0;
const float VREF = 5.0;
const float RANGE_MM = 30.0;
float zeroVoltage = 0.0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {;}
  
  // Calibrate the sensor to establish the zero point
  calibrateSensor();
  
  // Send a one-time ready signal to the Python script
  Serial.println("Laser Ready");
}

void loop() {
  int rawValue = analogRead(SENSOR_PIN);
  float voltage = rawValue * (VREF / 1023.0);
  float displacement_mm = (voltage - zeroVoltage) * (RANGE_MM / VREF);
  
  Serial.print("Dist_mm:");
  Serial.println(displacement_mm, 2);
  
  delay(10); // Approx. 100Hz update rate
}

void calibrateSensor() {
  long sum = 0;
  int numSamples = 100;
  delay(50);
  for (int i = 0; i < numSamples; i++) {
    sum += analogRead(SENSOR_PIN);
    delay(10);
  }
  float averageRaw = sum / (float)numSamples;
  zeroVoltage = averageRaw * (VREF / 1023.0);
}