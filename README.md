# Video Emotion Analysis

Real-time facial emotion recognition using WebSockets, OpenCV, and DeepFace.

## Features

- WebSocket server that receives video frames
- Facial emotion detection using DeepFace
- Auto-detection of video resolution
- Frame skipping for performance optimization
- Summary report with dominant emotion analysis

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and adjust settings:

```bash
cp .env.example .env
```

Available settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `WS_HOST` | `0.0.0.0` | WebSocket server host |
| `WS_PORT` | `8765` | WebSocket server port |
| `OUTPUT_FILE` | `emotion_results.json` | Output file for results |
| `ANALYZE_EVERY_N_FRAMES` | `1` | Analyze every Nth frame (higher = faster) |

## Usage

### Start the Server

```bash
python main.py
```

### Test with Webcam

```bash
python test_client.py
```

Press `q` to quit.

### Test with Video File

```bash
python test_video.py /path/to/video.mp4
```

## Protocol

1. Client connects to WebSocket server
2. Client sends JSON with frame dimensions: `{"width": 640, "height": 480}`
3. Server responds: `{"status": "ready"}`
4. Client sends raw BGR frame bytes
5. Server responds with emotion analysis:
   ```json
   {"status": "success", "faces_detected": 1, "emotions": ["happy"]}
   ```
6. On disconnect, server saves summary to `emotion_results.json`

## Output

After analysis, results are saved as JSON:

```json
{
  "total_frames_analyzed": 100,
  "emotion_frequency": {
    "happy": {"count": 60, "percentage": 60.0},
    "neutral": {"count": 30, "percentage": 30.0},
    "surprise": {"count": 10, "percentage": 10.0}
  },
  "average_scores": {
    "happy": 45.2,
    "neutral": 25.1,
    "sad": 10.5,
    "fear": 8.2,
    "angry": 5.0,
    "surprise": 4.0,
    "disgust": 2.0
  },
  "final_result": {
    "emotion": "happy",
    "confidence": 60.0
  }
}
```

## Performance Tips

Set `ANALYZE_EVERY_N_FRAMES=5` in `.env` to analyze every 5th frame, making processing ~5x faster while still capturing emotion changes.

## Based On

[Facial-Emotion-Recognition-using-OpenCV-and-Deepface](https://github.com/manish-9245/Facial-Emotion-Recognition-using-OpenCV-and-Deepface)
