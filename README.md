# Video Emotion Analysis

Real-time facial emotion recognition using Hume AI via WebSocket.

## Features

- Real-time WebSocket server for production use
- 48 distinct emotion dimensions
- Per-frame emotion detection
- Session summary with dominant emotion

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Get an API key from [Hume AI](https://www.hume.ai/)
2. Create `.env` file:

```bash
cp .env.example .env
```

3. Add your API key to `.env`:

```
HUME_API_KEY=your_api_key_here
```

## Usage

Start the server:

```bash
python server.py
```

The server listens on `ws://0.0.0.0:8765` by default. Configure via environment variables:

```
WS_HOST=0.0.0.0
WS_PORT=8765
ANALYZE_EVERY_N_FRAMES=5  # Analyze every 5th frame (faster)
```

**Testing the server:**

```bash
# Terminal 1: Start server
python server.py

# Terminal 2: Test with webcam (10 seconds default)
python test_client.py

# Webcam for 30 seconds
python test_client.py 30

# Or test with a video file
python test_client.py video.mp4
```

## Protocol

1. Connect to the WebSocket server
2. Receive `{"status": "ready"}` when connected
3. Send JPEG frames as binary data
4. Receive per-frame results:
   ```json
   {"status": "frame_processed", "frame": 5, "second": 2, "faces_detected": 1, "emotions": ["Joy"]}
   ```
   Or if frame was skipped (based on `ANALYZE_EVERY_N_FRAMES`):
   ```json
   {"status": "frame_skipped", "frame": 1}
   ```
5. Send `{"action": "end"}` to finish
6. Receive final summary:
   ```json
   {
     "status": "complete",
     "session_id": "abc123",
     "total_frames": 100,
     "frames_with_faces": 19,
     "processing_time_seconds": 5.2,
     "emotion_frequency": {"Joy": {"count": 12, "percentage": 63.2}},
     "average_scores": {"Joy": 0.45, "Calmness": 0.32},
     "timeline": {"0": "Calmness", "1": "Joy", "2": "Joy", "3": "Surprise", "4": "Joy"},
     "final_result": {"emotion": "Joy", "confidence": 63.2}
   }
   ```

   The `timeline` field contains the dominant emotion for each second of the session.

## Hume AI Emotions

Hume AI detects 48 emotions including:
- Joy, Sadness, Anger, Fear, Surprise, Disgust
- Admiration, Amusement, Anxiety, Awe
- Boredom, Calmness, Concentration, Confusion
- Contempt, Contentment, Craving, Determination
- And many more...

## Links

- [Hume AI](https://www.hume.ai/)
- [Hume API Docs](https://dev.hume.ai/)
