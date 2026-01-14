# Ascon-Lightweight-Cryptography

A secure AIoT (Artificial Intelligence of Things) implementation using ASCON-128a lightweight cryptography for encrypted split neural network inference on resource-constrained devices.

## Overview

This project demonstrates the integration of ASCON-128a authenticated encryption with associated data (AEAD) in an AIoT system for smoke detection. The system performs split inference where a neural network model is divided between a microcontroller (Arduino) and a server, with cryptographic operations ensuring data security during transmission.

## Features

- **Lightweight Cryptography**: ASCON-128a implementation optimized for constrained devices
- **Split Inference**: Neural network computation distributed between edge device and server
- **MQTT Communication**: Secure data transmission using ThingsBoard IoT platform
- **Real-time Processing**: Sensor data collection and encrypted inference on Arduino
- **Smoke Detection**: ANN-based classification using environmental sensor data

## Architecture

The system follows a client-server architecture:

1. **Client (Arduino)**: Runs `main.ino` - collects sensor data, performs partial neural network inference, encrypts results with ASCON-128a
2. **Server**: Runs `server.cpp` - receives encrypted data via MQTT, decrypts, completes inference, and sends results
3. **Communication**: MQTT protocol for secure data transmission between client and server

### Data Flow
```
Sensors → Client (Arduino - Split Inference + Encryption) → MQTT → Server (C++ - Decryption + Final Inference) → Results
```

## Demo

Watch the demonstration video showcasing the complete system in action:

[View Demo Video](demo/demoCryptoGraphy.mp4)

*Note: The video demonstrates the encrypted split inference process with real-time sensor data processing.*

## Installation

### Prerequisites

- Arduino IDE with ESP32/ESP8266 support
- C++ compiler for server (GCC/Clang)
- MQTT broker (ThingsBoard Cloud or local instance)
- Required Arduino libraries: WiFi, PubSubClient, ArduinoJson

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Ascon-Lightweight-Cryptography.git
   cd Ascon-Lightweight-Cryptography
   ```

2. **Configure Arduino:**
   - Open `main.ino` in Arduino IDE
   - Update WiFi credentials and MQTT settings
   - Upload to your microcontroller

3. **Setup Server:**
   - Configure server credentials in `server.cpp`
   - Compile and run the server:
     ```bash
     g++ server.cpp -o server
     ./server
     ```

4. **Model Training:**
   - Navigate to `src/model/`
   - Run data preprocessing:
     ```bash
     python handle_data.py
     ```
   - Train the model:
     ```bash
     python train.py
     ```
   - Extract parameters:
     ```bash
     python extract_parameters.py
     ```

## Usage

1. **Start the Server:**
   - Run the compiled server executable:
     ```bash
     ./server
     ```

2. **Run the Client:**
   - Upload `main.ino` to your Arduino device
   - Power on the Arduino device with sensors connected
   - Ensure MQTT connection is established

3. **System Operation:**
   The system will automatically:
   - Client collects sensor readings and performs partial inference
   - Client encrypts intermediate results with ASCON-128a
   - Encrypted data is transmitted via MQTT to the server
   - Server decrypts data, completes inference, and sends results back

## Technical Details

### ASCON-128a Configuration
- **Key Size**: 128 bits
- **Nonce Size**: 128 bits
- **Tag Size**: 128 bits
- **Block Size**: 64 bits

### Neural Network Architecture
- Input: 6 environmental parameters
- Hidden Layer: 4 neurons (quantized)
- Output: Binary classification (smoke detection)

### Performance Optimizations
- Quantized neural network weights for reduced memory usage
- Optimized ASCON implementation for microcontrollers
- Reduced cryptographic operations through split inference

## Future Work

- Implement multi-device coordination
- Extend support for additional cryptographic primitives
- Optimize for ultra-low power consumption
- Add support for federated learning
- Enhance security with key management protocols

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms specified in the LICENSE file.

## Contact

For questions or collaboration opportunities:
- Email: Anh.NT233258@sis.hust.edu.vn

---

*Project for Theory of Cryptography course* 
