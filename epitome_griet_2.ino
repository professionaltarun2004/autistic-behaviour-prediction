#include <WiFi.h>
#include <FirebaseESP32.h>
#include <Wire.h>
#include "MAX30105.h"

#define WIFI_SSID "Tarun"
#define WIFI_PASSWORD "Botarun@123"

#define FIREBASE_HOST "joystick-dc535-default-rtdb.asia-southeast1.firebasedatabase.app"
#define FIREBASE_AUTH "EnMSmfIoMO1D8mawx8y5mtarvpT1bHd6seAbL1QO"

FirebaseData firebaseData;
FirebaseAuth auth;
FirebaseConfig config;

#define VRX_PIN 34  // Joystick X-axis
#define VRY_PIN 35  // Joystick Y-axis
#define SW_PIN  32  // Joystick Switch

MAX30105 particleSensor;

const byte RATE_SIZE = 4; // Number of readings for averaging BPM
byte rates[RATE_SIZE];    // Stores last few BPM values
byte rateSpot = 0;
long lastBeat = 0;        // Timestamp of last beat
float beatsPerMinute;
int beatAvg;

void setup() {
    Serial.begin(115200);

    // Connect to WiFi
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected!");

    // Initialize Firebase
    config.host = FIREBASE_HOST;
    config.signer.tokens.legacy_token = FIREBASE_AUTH;
    Firebase.begin(&config, &auth);
    Firebase.reconnectWiFi(true);
    Firebase.setReadTimeout(firebaseData, 1000);
    Firebase.setwriteSizeLimit(firebaseData, "tiny");

    // Set up joystick pins
    pinMode(SW_PIN, INPUT_PULLUP);
    pinMode(VRX_PIN, INPUT);
    pinMode(VRY_PIN, INPUT);

    // Initialize MAX30102 sensor
    if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
        Serial.println("MAX30102 not found. Check connections!");
        while (1); // Halt execution if sensor is not found
    }
    Serial.println("MAX30102 detected!");

    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeGreen(0);
}

void loop() {
    // Read joystick values
    int x = analogRead(VRX_PIN);
    int y = analogRead(VRY_PIN);
    int sw = digitalRead(SW_PIN) == LOW ? 1 : 0;
    
    // Get heart rate
    int bpm = getBPM();

    // Update Firebase with joystick data
    if (Firebase.setInt(firebaseData, "/joystick/x", x)) {
        Serial.println("X updated: " + String(x));
    } else {
        Serial.println("Firebase X Error: " + firebaseData.errorReason());
    }

    if (Firebase.setInt(firebaseData, "/joystick/y", y)) {
        Serial.println("Y updated: " + String(y));
    } else {
        Serial.println("Firebase Y Error: " + firebaseData.errorReason());
    }

    if (Firebase.setInt(firebaseData, "/joystick/sw", sw)) {
        Serial.println("SW updated: " + String(sw));
    } else {
        Serial.println("Firebase SW Error: " + firebaseData.errorReason());
    }

    // Update Firebase with heart rate
    if (Firebase.setInt(firebaseData, "/heartRate/bpm", bpm)) {
        Serial.println("Heart Rate (BPM) updated: " + String(bpm));
    } else {
        Serial.println("Firebase BPM Error: " + firebaseData.errorReason());
    }

    delay(500);
}

// Function to calculate BPM
int getBPM() {
    long irValue = particleSensor.getIR();

    if (irValue < 50000) {  // No finger detected
        Serial.println("No finger detected.");
        return 0;
    }

    if (particleSensor.check() == true) {  // Check for new data
        if (particleSensor.getIR() > 50000) {  // Simple threshold-based detection
            long delta = millis() - lastBeat;
            lastBeat = millis();
            beatsPerMinute = 60 / (delta / 1000.0);

            if (beatsPerMinute > 50 && beatsPerMinute < 255) {  // Valid BPM range
                rates[rateSpot++] = (byte)beatsPerMinute;
                rateSpot %= RATE_SIZE;

                beatAvg = 0;
                for (byte i = 0; i < RATE_SIZE; i++) {
                    beatAvg += rates[i];
                }
                beatAvg /= RATE_SIZE;
            }
        }
    }

    return beatAvg;
}