import time
import requests

# ----------------------------
# ThingsBoard configuration
# ----------------------------
TB_HOST = "https://thingsboard.cloud"

# ⚠️ PUT YOUR OWN VALUES HERE
USERNAME = "Anh.NT233258@sis.hust.edu.vn"
PASSWORD = "G@hNFHtePV4ciXS"

# MUST be UUID, NOT device name
DEVICE_ID = "73dbaca0-eef5-11f0-bb6b-45643ceafb13"

# Telemetry keys you send from ESP32
KEYS = "cipher,tag"

POLL_INTERVAL = 3  # seconds

# ----------------------------
# Login & get JWT
# ----------------------------
def tb_login():
    url = f"{TB_HOST}/api/auth/login"
    resp = requests.post(url, json={
        "username": USERNAME,
        "password": PASSWORD
    })
    resp.raise_for_status()
    return resp.json()["token"]

# ----------------------------
# Fetch latest telemetry
# ----------------------------
def get_latest_telemetry(jwt):
    headers = {
        "X-Authorization": f"Bearer {jwt}"
    }

    url = (
        f"{TB_HOST}/api/plugins/telemetry/DEVICE/"
        f"{DEVICE_ID}/values/timeseries"
        f"?keys={KEYS}"
    )

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

# ----------------------------
# Main loop
# ----------------------------
def main():
    print("🔐 Logging into ThingsBoard...")
    jwt = tb_login()
    print("✅ Login OK")

    last_ts = None

    print("🚀 Listening for telemetry (REST polling)...")
    while True:
        data = get_latest_telemetry(jwt)

        if "cipher" in data and "tag" in data:
            cipher = data["cipher"][0]
            tag = data["tag"][0]

            ts = cipher["ts"]

            # Only print NEW data
            if ts != last_ts:
                last_ts = ts
                print("📩 NEW TELEMETRY")
                print("Timestamp:", ts)
                print("Cipher:", cipher["value"])
                print("Tag   :", tag["value"])
                print("-" * 40)

        time.sleep(POLL_INTERVAL)

# ----------------------------
if __name__ == "__main__":
    main()
