"""
Test client for the WebSocket emotion analysis server.
Usage:
  python test_client.py                  # Webcam for 10 seconds
  python test_client.py 30               # Webcam for 30 seconds
  python test_client.py video.mp4        # Video file
"""

import asyncio
import json
import sys

import cv2
import websockets


async def test_with_video(video_path: str, server_url: str = "ws://localhost:8765"):
    """Test server with a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Testing with video: {video_path} ({total_frames} frames)")
    print(f"Connecting to {server_url}...")

    async with websockets.connect(server_url) as ws:
        # Wait for ready
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"Server: {data}")

        if data.get("status") != "ready":
            print("Server not ready")
            return

        frame_count = 0
        skip_frames = 5  # Send every 5th frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame)
            jpeg_bytes = buffer.tobytes()

            # Send frame
            await ws.send(jpeg_bytes)

            # Get response
            response = await ws.recv()
            result = json.loads(response)

            if result.get("status") == "frame_processed":
                emotions = result.get("emotions", [])
                faces = result.get("faces_detected", 0)
                print(f"Frame {result['frame']}: {faces} face(s), emotions: {emotions}")
            elif result.get("status") == "error":
                print(f"Frame {result['frame']} error: {result.get('message')}")

        cap.release()

        # Send end signal
        await ws.send(json.dumps({"action": "end"}))

        # Get summary
        summary = await ws.recv()
        result = json.loads(summary)

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(json.dumps(result, indent=2))


async def test_with_webcam(server_url: str = "ws://localhost:8765", duration: int = 10):
    """Test server with webcam for specified duration."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print(f"Testing with webcam for {duration} seconds")
    print(f"Connecting to {server_url}...")

    async with websockets.connect(server_url) as ws:
        # Wait for ready
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"Server: {data}")

        if data.get("status") != "ready":
            print("Server not ready")
            return

        import time
        start_time = time.time()
        frame_count = 0

        print("Streaming webcam frames... Press Ctrl+C to stop early")

        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_count += 1

                # Skip frames for performance
                if frame_count % 5 != 0:
                    continue

                # Encode frame as JPEG
                _, buffer = cv2.imencode(".jpg", frame)
                jpeg_bytes = buffer.tobytes()

                # Send frame
                await ws.send(jpeg_bytes)

                # Get response
                response = await ws.recv()
                result = json.loads(response)

                if result.get("status") == "frame_processed":
                    emotions = result.get("emotions", [])
                    faces = result.get("faces_detected", 0)
                    print(f"Frame {result['frame']}: {faces} face(s), emotions: {emotions}")

        except KeyboardInterrupt:
            print("\nStopping...")

        cap.release()

        # Send end signal
        await ws.send(json.dumps({"action": "end"}))

        # Get summary
        summary = await ws.recv()
        result = json.loads(summary)

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # Check if arg is a number (duration) or a file path
        if arg.isdigit():
            asyncio.run(test_with_webcam(duration=int(arg)))
        else:
            asyncio.run(test_with_video(arg))
    else:
        asyncio.run(test_with_webcam())
