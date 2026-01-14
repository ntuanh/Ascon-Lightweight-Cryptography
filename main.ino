#include <Arduino.h>

#include "split_inference.h"
#include "ascon-128a.h"
#include "mqtt_connection.h"
#include "sensors.h"
#define NUM_MEASURE 1000


// --------------------
// WiFi / MQTT config
// --------------------
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASS = "";

const char* MQTT_SERVER = "thingsboard.cloud";
const int   MQTT_PORT   = 1883;
const char* ACCESS_TOKEN = "private";

// --------------------
// ASCON config
// --------------------
static uint8_t ASCON_KEY[16] = {0};   // demo key
static uint8_t ASCON_NONCE[16] = {0}; // demo nonce
static uint8_t ASCON_AD[1] = {0};     // no associated data

// --------------------
// Buffers
// --------------------
// float input_x[6] = {0.3f, 0.2f, -0.8f, 1.0f, -1.5f, 0.2f};
float input_x[6] ; 

float feature_map[4];
int   q_feature[4];
float scale;

uint8_t plaintext[32];
uint8_t ciphertext[32];
uint8_t tag[16];

uint32_t loop_times[NUM_MEASURE];
int loop_index = 0;

// --------------------
// Helpers
// --------------------
void pack_quantized_features(
    const int* q,
    int size,
    float scale,
    uint8_t* out,
    int& out_len
) {
    // format:
    // [scale(float)][q0][q1][q2][q3]
    memcpy(out, &scale, sizeof(float));
    for (int i = 0; i < size; i++) {
        out[sizeof(float) + i] = (uint8_t)(q[i] & 0xFF);
    }
    out_len = sizeof(float) + size;
}

String to_hex(const uint8_t* data, int len) {
    const char hexmap[] = "0123456789ABCDEF";
    String s;
    for (int i = 0; i < len; i++) {
        s += hexmap[(data[i] >> 4) & 0xF];
        s += hexmap[data[i] & 0xF];
    }
    return s;
}

// --------------------
// Arduino setup
// --------------------
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("=== ESP32 Split Inference + ASCON + MQTT ===");

    // Init WiFi + MQTT
    mqtt_init(
        WIFI_SSID,
        WIFI_PASS,
        MQTT_SERVER,
        MQTT_PORT,
        ACCESS_TOKEN
    );

    sensors_init();

    get_data_sensors_scaled(input_x);
    

}

// --------------------
// Arduino loop
// --------------------
void loop() {
  // Stop after 20 measurements
    if (loop_index >= NUM_MEASURE) {

        Serial.println("\n⏱ LOOP TIME RESULTS (ms)");
        Serial.print("[ ");

        for (int i = 0; i < NUM_MEASURE; i++) {
            Serial.print(loop_times[i]);
            if (i < NUM_MEASURE - 1) Serial.print(", ");
        }

        Serial.println(" ]");

        // Stop forever
        while (true) {
            delay(1000);
        }
    }
    uint32_t t_start = millis();

    get_data_sensors_scaled(input_x);
    // =================================================
    // 1. Dummy input → split inference
    // =================================================
    forward_3layers(input_x, feature_map);

    // =================================================
    // 2. Quantization (4 bits)
    // =================================================
    quantize_nbits(feature_map, q_feature, 4, 4, scale);

    // =================================================
    // 3. Pack data for encryption
    // =================================================
    int plen = 0;
    pack_quantized_features(q_feature, 4, scale, plaintext, plen);

    // =================================================
    // 4. ASCON-128a encryption
    // =================================================
    ascon128a_encrypt(
        ciphertext,
        tag,
        plaintext,
        plen,
        ASCON_AD,
        0,
        ASCON_NONCE,
        ASCON_KEY
    );

    // =================================================
    // 5. Build JSON payload
    // =================================================
    String payload = "{";
    payload += "\"cipher\":\"" + to_hex(ciphertext, plen) + "\",";
    payload += "\"tag\":\"" + to_hex(tag, 16) + "\"";
    // payload += "\"time\":\"" + String(t_start_ms) + "\"";
    payload += "}";

    // =================================================
    // 6. Publish to ThingsBoard
    // =================================================
    mqtt_publish("v1/devices/me/telemetry", payload.c_str());
    mqtt_loop();

    // =================================================
    // Debug output
    // =================================================
    Serial.println("Published encrypted feature map:");
    Serial.println(payload);

    uint32_t t_end = millis();

    // Save measurement
    loop_times[loop_index] = t_end - t_start;

    Serial.print("Loop ");
    Serial.print(loop_index);
    Serial.print(" time = ");
    Serial.print(loop_times[loop_index]);
    Serial.println(" ms");

    // loop_index++;

    delay(2000);
}
