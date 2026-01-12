#ifndef MQTT_CONNECTION_H
#define MQTT_CONNECTION_H

#include <WiFi.h>
#include <PubSubClient.h>

// --------------------
// WiFi + MQTT config
// --------------------
void mqtt_init(
    const char* ssid,
    const char* password,
    const char* mqtt_server,
    int mqtt_port,
    const char* access_token
);

// --------------------
// MQTT loop handler
// --------------------
void mqtt_loop();

// --------------------
// Publish telemetry
// --------------------
bool mqtt_publish(const char* topic, const char* payload);

#endif
