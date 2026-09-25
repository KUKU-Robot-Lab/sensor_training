// X, Y, Z 축 핀 정의
#define X_PUL 3
#define X_DIR 4
#define X_ENA 5

#define Y_PUL 6
#define Y_DIR 7
#define Y_ENA 8

#define Z_PUL 9
#define Z_DIR 10
#define Z_ENA 11

void setup() {
  Serial.begin(115200);

  // 핀 모드 설정
  pinMode(X_PUL, OUTPUT); pinMode(X_DIR, OUTPUT); pinMode(X_ENA, OUTPUT);
  pinMode(Y_PUL, OUTPUT); pinMode(Y_DIR, OUTPUT); pinMode(Y_ENA, OUTPUT);
  pinMode(Z_PUL, OUTPUT); pinMode(Z_DIR, OUTPUT); pinMode(Z_ENA, OUTPUT);

  // 모든 드라이버 활성화
  digitalWrite(X_ENA, LOW);
  digitalWrite(Y_ENA, LOW);
  digitalWrite(Z_ENA, LOW);

  Serial.println("3-Axis Motor control firmware ready.");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    processCommand(cmd);
  }
}

void processCommand(String cmd) {
  char commandType = cmd.charAt(0);

  if (commandType == 'M') {
    // 형식: M,axis,steps,delay
    int sep1 = cmd.indexOf(',');
    int sep2 = cmd.indexOf(',', sep1 + 1);
    int sep3 = cmd.indexOf(',', sep2 + 1);

    if (sep1 < 0 || sep2 < 0 || sep3 < 0) return;

    char axis = cmd.charAt(sep1 + 1);
    long steps = cmd.substring(sep2 + 1, sep3).toInt();
    int move_delay = cmd.substring(sep3 + 1).toInt();

    moveMotor(axis, steps, move_delay);

  } else if (commandType == 'S') {
    // 모든 모터 정지 (ENA 핀을 HIGH로 설정하여 비활성화)
    digitalWrite(X_ENA, HIGH);
    digitalWrite(Y_ENA, HIGH);
    digitalWrite(Z_ENA, HIGH);
  }
}

void moveMotor(char axis, long steps, int move_delay) {
  int pin_pul, pin_dir;

  if (axis == 'X') {
    pin_pul = X_PUL;
    pin_dir = X_DIR;
  } else if (axis == 'Y') {
    pin_pul = Y_PUL;
    pin_dir = Y_DIR;
  } else if (axis == 'Z') {
    pin_pul = Z_PUL;
    pin_dir = Z_DIR;
  } else {
    return; // 지원하지 않는 축
  }

  // 방향 설정 (steps가 양수면 HIGH, 음수면 LOW로 가정)
  digitalWrite(pin_dir, steps > 0);

  for (long i = 0; i < abs(steps); i++) {
    digitalWrite(pin_pul, HIGH);
    delayMicroseconds(2);
    digitalWrite(pin_pul, LOW);
    delayMicroseconds(move_delay);
  }
}