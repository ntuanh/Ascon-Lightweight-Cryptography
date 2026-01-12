#include "mqtt_connection.h"

// --------------------
// Internal objects
// --------------------
static WiFiClient espClient;
static PubSubClient client(espClient);

// --------------------
// Init WiFi + MQTT
// --------------------
void mqtt_init(
    const char* ssid,
    const char* password,
    const char* mqtt_server,
    int mqtt_port,
    const char* access_token
) {
    // WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
    }

    // MQTT
    client.setServer(mqtt_server, mqtt_port);

    while (!client.connected()) {
        client.connect("ESP32", access_token, NULL);
        delay(500);
    }
}

// --------------------
// MQTT loop
// --------------------
void mqtt_loop() {
    if (!client.connected()) {
        // Reconnect logic (simple)
        while (!client.connected()) {
            client.connect("ESP32");
            delay(500);
        }
    }
    client.loop();
}

// --------------------
// Publish telemetry
// --------------------
bool mqtt_publish(const char* topic, const char* payload) {
    if (!client.connected()) return false;
    return client.publish(topic, payload);
}
