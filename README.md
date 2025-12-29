# Parakeet Whisper API (ONNX INT8 Backend)

A high-performance, OpenAI-compatible local speech transcription service using the **Parakeet TDT 1.1B** model (via ONNX Runtime INT8).

## 🚀 Features

-   **Super Fast**: ~17x faster than real-time on standard GPU/CPU setups using ONNX INT8 quantization.
-   **Low Memory**: Uses quantization to reduce VRAM/RAM usage.
-   **OpenAI Compatible**: Drop-in replacement for OpenAI's `/v1/audio/transcriptions` endpoint.
-   **Web Interface**: Simple drag-and-drop UI for easy testing.
-   **Sanitized Output**: Automatically improved spacing and punctuation.

## 📊 Benchmarks

Real-world performance tested on `story_spanish.mp3`:
-   **Audio Duration**: 41.62s
-   **Processing Time**: ~2.35s
-   **Speedup**: **17.6x** faster than real-time ⚡
-   **Real Time Factor (RTF)**: **0.057**

## 🛠️ Installation

1.  **Environment**: ensure you have Conda installed.
    ```bash
    conda create -n parakeet-onnx python=3.10
    conda activate parakeet-onnx
    pip install -r requirements.txt
    ```

2.  **Dependencies**:
    -   `ffmpeg` must be installed on your system.

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
