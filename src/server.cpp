#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <string>
#include <unistd.h>


#include <curl/curl.h>
#include <nlohmann/json.hpp>             // nlohmann::json
#include "ascon-128a.h"
#include "nn_weights.h"


using json = nlohmann::json;

// ==========================
// ThingsBoard config
// ==========================
static const std::string TB_HOST   = "https://thingsboard.cloud";
static const std::string USERNAME  = "Anh.NT233258@sis.hust.edu.vn";
static const std::string PASSWORD  = "G@hNFHtePV4ciXS";
static const std::string DEVICE_ID = "73dbaca0-eef5-11f0-bb6b-45643ceafb13";

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

    while (true) {
        std::string cipher_hex, tag_hex;

        std::cout << " Waiting for telemetry...\n";

        if (!get_latest_cipher_tag(jwt, cipher_hex, tag_hex)) {
            std::cerr << "No telemetry yet\n\n";
            sleep(2);
            continue;
        }

        std::cout << "Telemetry received\n";
        std::cout << "Cipher (hex): " << cipher_hex << "\n";
        std::cout << "Tag    (hex): " << tag_hex << "\n";

        // --------------------------
        // Decode hex
        // --------------------------
        auto C = hex_to_bytes(cipher_hex);
        auto T = hex_to_bytes(tag_hex);
        std::vector<uint8_t> P(C.size());

        // --------------------------
        // ASCON decrypt
        // --------------------------
        std::cout << "Decrypting (ASCON-128a)...\n";

        bool ok = ascon128a_decrypt(
            P.data(),
            C.data(), C.size(),
            ASCON_AD, 0,
            ASCON_NONCE,
            ASCON_KEY,
            T.data()
        );

        if (!ok) {
            std::cerr << "AUTH FAILED (tag mismatch)\n\n";
            continue;
        }

        std::cout << "Decryption Done\n";

        // --------------------------
        // Unpack quantized payload
        // --------------------------
        float scale;
        memcpy(&scale, P.data(), sizeof(float));

        int q[4];
        for (int i = 0; i < 4; i++)
            q[i] = (int8_t)P[sizeof(float) + i];

        float features[4];
        for (int i = 0; i < 4; i++)
            features[i] = q[i] / scale;

        // std::cout << "\n Recovered feature map\n";
        // std::cout << "Scale = " << scale << "\n";
        // for (int i = 0; i < 4; i++)
        //     std::cout << "Feature[" << i << "] = " << features[i] << "\n";

        // --------------------------
        // Forward last layer
        // --------------------------
        float logits[2];
        forward_last_layer(features, logits);

        // Argmax
        int pred = (logits[0] > logits[1]) ? 0 : 1;

        // --------------------------
        // Final output
        // --------------------------
        // std::cout << "\n=====================================\n";
        // std::cout << "FINAL MODEL OUTPUT\n";
        // std::cout << "=====================================\n";
        // std::cout << "Logit[0] = " << logits[0] << "\n";
        // std::cout << "Logit[1] = " << logits[1] << "\n";
        float res = logits[1] + 2*logits[0];
        std::cout << "res = " << res << "\n";
        
        if (res > 3.3) {
            std::cout << "Fire !!!\n";
        }
        else {
            std::cout << "No Fire\n";
        }

        sleep(2);
    }
}
