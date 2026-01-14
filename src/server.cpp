#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <string>
#include <unistd.h>
#include <chrono>


#include <curl/curl.h>
#include <nlohmann/json.hpp>             // nlohmann::json
#include "ascon-128a.h"
#include "nn_weights.h"

#define NUM_MEASURE 10000



using json = nlohmann::json;

// ==========================
// ThingsBoard config
// ==========================
static const std::string TB_HOST   = "https://thingsboard.cloud";
static const std::string USERNAME  = "Anh.NT233258@sis.hust.edu.vn";
static const std::string PASSWORD  = "private";
static const std::string DEVICE_ID = "private";

// ==========================
// ASCON config (MUST MATCH ESP32)
// ==========================
static uint8_t ASCON_KEY[16]   = {0};
static uint8_t ASCON_NONCE[16] = {0};
static uint8_t ASCON_AD[1]     = {0};

// ==========================
// CURL helpers
// ==========================
static size_t write_cb(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::string http_post(const std::string& url, const std::string& body) {
    CURL* curl = curl_easy_init();
    std::string response;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return response;
}

std::string http_get(const std::string& url, const std::string& token) {
    CURL* curl = curl_easy_init();
    std::string response;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("X-Authorization: Bearer " + token).c_str());

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return response;
}

// ==========================
// ThingsBoard API
// ==========================
std::string tb_login() {
    std::string body =
        "{\"username\":\"" + USERNAME + "\",\"password\":\"" + PASSWORD + "\"}";

    auto resp = http_post(TB_HOST + "/api/auth/login", body);
    auto j = json::parse(resp);
    return j["token"];
}

bool get_latest_cipher_tag(
    const std::string& jwt,
    std::string& cipher_hex,
    std::string& tag_hex
) {
    std::string url =
        TB_HOST + "/api/plugins/telemetry/DEVICE/" +
        DEVICE_ID + "/values/timeseries?keys=cipher,tag";

    auto resp = http_get(url, jwt);
    auto j = json::parse(resp);

    if (!j.contains("cipher") || !j.contains("tag")) return false;

    cipher_hex = j["cipher"][0]["value"];
    tag_hex    = j["tag"][0]["value"];
    return true;
}

// ==========================
// Utilities
// ==========================
std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> out;
    for (size_t i = 0; i < hex.length(); i += 2) {
        out.push_back(std::stoi(hex.substr(i, 2), nullptr, 16));
    }
    return out;
}


void forward_last_layer(const float in[4], float out[2]) {
    for (int i = 0; i < 2; i++) {
        float sum = b3[i];
        for (int j = 0; j < 4; j++) {
            sum += W3[i][j] * in[j];
        }
        out[i] = sum;
    }
}
int main() {
    curl_global_init(CURL_GLOBAL_ALL);

    std::cout << "Logging into ThingsBoard...\n";
    std::string jwt = tb_login();
    std::cout << "Login Done\n\n";

    using Clock = std::chrono::high_resolution_clock;
    using ms = std::chrono::duration<double, std::milli>;

    std::vector<double> loop_times;
    loop_times.reserve(NUM_MEASURE);

    int count = 0;

    while (count < NUM_MEASURE) {

        std::string cipher_hex, tag_hex;
        std::cout << "Waiting for telemetry...\n";

        if (!get_latest_cipher_tag(jwt, cipher_hex, tag_hex)) {
            std::cerr << "No telemetry yet\n";
            sleep(2);
            continue;
        }

        // =========================
        // START timing
        // =========================
        auto t_start = Clock::now();

        // --------------------------
        // Decode hex
        // --------------------------
        auto C = hex_to_bytes(cipher_hex);
        auto T = hex_to_bytes(tag_hex);
        std::vector<uint8_t> P(C.size());

        // --------------------------
        // ASCON decrypt
        // --------------------------
        bool ok = ascon128a_decrypt(
            P.data(),
            C.data(), C.size(),
            ASCON_AD, 0,
            ASCON_NONCE,
            ASCON_KEY,
            T.data()
        );

        if (!ok) {
            std::cerr << "AUTH FAILED\n";
            continue;
        }

        // --------------------------
        // Dequant + inference
        // --------------------------
        float scale;
        memcpy(&scale, P.data(), sizeof(float));

        int q[4];
        for (int i = 0; i < 4; i++)
            q[i] = (int8_t)P[sizeof(float) + i];

        float features[4];
        for (int i = 0; i < 4; i++)
            features[i] = q[i] / scale;

        float logits[2];
        forward_last_layer(features, logits);

        int pred = (logits[0] > logits[1]) ? 0 : 1;

        float res = logits[1] + 2 * logits[0];

        // =========================
        // END timing
        // =========================
        auto t_end = Clock::now();

        double elapsed_ms = ms(t_end - t_start).count();
        loop_times.push_back(elapsed_ms);

        // =========================
        // Print per-iteration result
        // =========================
        std::cout << "Loop " << count
                  << " time = " << elapsed_ms << " ms | ";

        if (res > 3.3)
            std::cout << "Fire !!!\n";
        else
            std::cout << "No Fire\n";

        // count++;
        sleep(2);
    }

    // =========================
    // Print all results
    // =========================
    std::cout << "\nSERVER LOOP TIMES (ms)\n[ ";
    for (size_t i = 0; i < loop_times.size(); i++) {
        std::cout << loop_times[i];
        if (i < loop_times.size() - 1) std::cout << ", ";
    }
    std::cout << " ]\n";

    return 0;
}
