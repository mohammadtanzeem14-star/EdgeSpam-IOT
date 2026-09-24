#include "model_weights.h"
#include <math.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define NUM_FEATURES 2048

// OLED
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDRESS 0x3C

Adafruit_SSD1306 display(
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  &Wire,
  OLED_RESET
);


// --------------------------------------------------
// MurmurHash3
// --------------------------------------------------
uint32_t murmurHash3(const char *data, int len) {

  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;

  uint32_t h1 = 0;

  int roundedEnd = len & 0xfffffffc;

  for (int i = 0; i < roundedEnd; i += 4) {

    uint32_t k1 =
      ((uint32_t)data[i] & 0xff) |
      (((uint32_t)data[i + 1] & 0xff) << 8) |
      (((uint32_t)data[i + 2] & 0xff) << 16) |
      (((uint32_t)data[i + 3] & 0xff) << 24);

    k1 *= c1;
    k1 = (k1 << 15) | (k1 >> 17);
    k1 *= c2;

    h1 ^= k1;

    h1 = (h1 << 13) | (h1 >> 19);
    h1 = h1 * 5 + 0xe6546b64;
  }

  uint32_t k1 = 0;

  switch (len & 3) {

    case 3:
      k1 ^= ((uint32_t)data[roundedEnd + 2] & 0xff) << 16;

    case 2:
      k1 ^= ((uint32_t)data[roundedEnd + 1] & 0xff) << 8;

    case 1:
      k1 ^= ((uint32_t)data[roundedEnd] & 0xff);

      k1 *= c1;
      k1 = (k1 << 15) | (k1 >> 17);
      k1 *= c2;

      h1 ^= k1;
  }

  h1 ^= len;

  h1 ^= h1 >> 16;
  h1 *= 0x85ebca6b;
  h1 ^= h1 >> 13;
  h1 *= 0xc2b2ae35;
  h1 ^= h1 >> 16;

  return h1;
}


// --------------------------------------------------
// Create Features
// --------------------------------------------------
void createFeatures(String message, float features[]) {

  for (int i = 0; i < NUM_FEATURES; i++) {
    features[i] = 0.0f;
  }

  message.toLowerCase();

  char token[50];
  int tokenLength = 0;

  for (int i = 0; i <= message.length(); i++) {

    char c;

    if (i < message.length()) {
      c = message[i];
    } else {
      c = ' ';
    }

    bool valid =
      ((c >= 'a' && c <= 'z') ||
       (c >= '0' && c <= '9') ||
       c == '_');

    if (valid) {

      if (tokenLength < 49) {
        token[tokenLength++] = c;
      }

    } else {

      if (tokenLength >= 2) {

        token[tokenLength] = '\0';

        uint32_t hashValue =
          murmurHash3(token, tokenLength);

        int index =
          hashValue % NUM_FEATURES;

        features[index] += 1.0f;
      }

      tokenLength = 0;
    }
  }

  // L2 normalization
  float sumSquares = 0.0f;

  for (int i = 0; i < NUM_FEATURES; i++) {
    sumSquares += features[i] * features[i];
  }

  float norm = sqrt(sumSquares);

  if (norm > 0.0f) {

    for (int i = 0; i < NUM_FEATURES; i++) {
      features[i] /= norm;
    }
  }
}


// --------------------------------------------------
// ML Score
// --------------------------------------------------
float calculateScore(float features[]) {

  float score = MODEL_BIAS;

  for (int i = 0; i < NUM_FEATURES; i++) {

    if (features[i] != 0.0f) {
      score += MODEL_WEIGHTS[i] * features[i];
    }
  }

  return score;
}


// --------------------------------------------------
// OLED Display
// --------------------------------------------------
void showResult(String result, float score) {

  display.clearDisplay();

  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);

  display.println("EdgeSpam-IoT");

  display.drawLine(
    0, 12,
    127, 12,
    SSD1306_WHITE
  );

  display.setTextSize(2);
  display.setCursor(0, 22);

  display.println(result);

  display.setTextSize(1);
  display.setCursor(0, 48);

  display.print("Score: ");
  display.println(score, 2);

  display.display();
}


// --------------------------------------------------
// Setup
// --------------------------------------------------
void setup() {

  Serial.begin(115200);

  delay(1000);

  // Start I2C
  Wire.begin(21, 22);

  // Start OLED
  if (!display.begin(
        SSD1306_SWITCHCAPVCC,
        OLED_ADDRESS
      )) {

    Serial.println("OLED not found!");

  } else {

    display.clearDisplay();

    display.setTextColor(SSD1306_WHITE);

    display.setTextSize(1);
    display.setCursor(0, 0);

    display.println("EdgeSpam-IoT");

    display.println();
    display.println("OLED Ready");

    display.display();

    delay(2000);
  }

  Serial.println();
  Serial.println("================================");
  Serial.println("       EdgeSpam-IoT");
  Serial.println("================================");
  Serial.println("Type an SMS and press Enter.");
  Serial.println();
}


// --------------------------------------------------
// Main Loop
// --------------------------------------------------
void loop() {

  if (Serial.available()) {

    String message =
      Serial.readStringUntil('\n');

    message.trim();

    if (message.length() == 0) {
      return;
    }

    static float features[NUM_FEATURES];

    createFeatures(message, features);

    float score =
      calculateScore(features);

    String result;

    if (score >= 0.0f) {

      result = "SPAM";

    } else {

      result = "HAM";
    }

    // Serial output
    Serial.println();
    Serial.println("--- RESULT ---");

    Serial.print("SMS: ");
    Serial.println(message);

    Serial.print("Score: ");
    Serial.println(score, 6);

    Serial.print("Classification: ");
    Serial.println(result);

    Serial.println("---------------");

    // OLED output
    showResult(result, score);
  }
}