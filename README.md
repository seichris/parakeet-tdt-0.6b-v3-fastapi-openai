# Parakeet Openai API Compatible (ONNX INT8 Backend)

A high-performance, OpenAI-compatible local speech transcription service using the **Parakeet TDT 1.1B** model (via ONNX Runtime INT8).

## 🚀 Features

-   **Super Fast**: ~17x faster than real-time on  CPU setups using ONNX INT8 quantization, way faster than Whisper Large turbo v3.
-   **Low Memory**: Uses quantization to reduce VRAM/RAM usage. (Consumes around 6500mb of ram)
-   **OpenAI Compatible**: Drop-in replacement for OpenAI's `/v1/audio/transcriptions` endpoint.
-   **Web Interface**: Simple drag-and-drop UI for easy testing.
-   **Sanitized Output**: Automatically improved spacing and punctuation.

## 📊 Benchmarks

Real-world performance tested on **Parakeet TDT 1.1B (ONNX INT8)**.

### Speed Comparison

| CPU | Audio File | Duration | Processing Time | Speedup | RTF |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **i7 12700KF** 🚀 | `story_spanish.mp3` | 41.62s | **1.40s** | **29.7x** | **0.033** |
| i7 4790 | `story_spanish.mp3` | 41.62s | ~2.35s | 17.6x | 0.057 |

### General Performance (i7 12700KF)

Measured with **P-core Optimization (Threads=8, Pinned)**:
- **Average Speedup**: **~29x** ⚡
- **Real Time Factor**: **0.033**
- **Processing**: ~0.3s per 10s chunk

*Note: Benchmarks run using local endpoint with INT8 quantization, pinned to P-cores.*

## ⚡ Performance Optimization Guide

To achieve the **29.7x speedup** on hybrid CPUs (like Intel 12th/13th/14th Gen), it is critical to use **P-cores only**.

### Linux (Recommended)
Use `taskset` to pin the process to Performance cores (usually 0-15 on i7-12700K).

```bash
# Example for 8 P-cores (cores 0-7 physical, 8-15 hyperthreads)
taskset -c 0-15 python app.py
```

### Resource Usage
- **RAM**: ~3.0 GB (Model + Runtime)
- **CPU**: 100% usage on assigned cores during transcription
- **VRAM**: 0 GB (Runs entirely on CPU)

## 🛠️ Installation

### 1. Prerequisites
Ensure you have **Python 3.10** and **FFmpeg** installed.

#### Linux (Ubuntu/Debian)
```bash
sudo apt update && sudo apt install ffmpeg
```

#### Conda (Recommended)
```bash
conda create -n parakeet-onnx python=3.10
conda activate parakeet-onnx
```

### 2. Install Dependencies
Clone this repository and install the required packages:
```bash
git clone https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai
cd parakeet-tdt-0.6b-v3-fastapi-openai
pip install -r requirements.txt
```

## 🧠 Supported Models

The API accepts the following model names in the `model` parameter:
-   `whisper-1` (Default)
-   `parakeet`
-   `parakeet-tdt-0.6b-v3`

Both behave identically, processing audio with the **Parakeet TDT 1.1B (ONNX INT8)** model.

## 🚀 Usage

### Start the Server
```bash
conda activate parakeet-onnx
python app.py
```
*Port: 5092 (Default)*

### Web Interface
Open [http://127.0.0.1:5092](http://127.0.0.1:5092) in your browser.

## 🔌 API Documentation

This server provides an OpenAI-compatible endpoint.

### Endpoint
`POST /v1/audio/transcriptions`

### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | `file` | **Required** | Audio file (mp3, wav, m4a, etc.) |
| `model` | `string` | `whisper-1` | Model name (accepted but ignored) |
| `response_format` | `string` | `json` | Output format: `json`, `text`, `srt`, `vtt`, `verbose_json` |

### Examples

**Python (openai-python)**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5092/v1",
    api_key="sk-any"
)

audio_file = open("audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file,
  response_format="text"
)
print(transcript)
```

**cURL**
```bash
curl http://127.0.0.1:5092/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

## 📂 Project Structure
-   `app.py`: Main server application (Flask + Waitress).
-   `models/`: Directory where ONNX models are stored (symlinked).
-   `templates/`: HTML frontend templates.
-   `requirements.txt`: Python dependencies.
