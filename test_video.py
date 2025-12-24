import asyncio
import json
import sys
import cv2
import websockets


async def send_video(video_path: str):
    uri = "ws://localhost:8765"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Get video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps

    print(f"Video: {frame_width}x{frame_height} @ {fps:.1f}fps")
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri, max_size=10 * 1024 * 1024) as ws:
        # Send dimensions first
        await ws.send(json.dumps({"width": frame_width, "height": frame_height}))
        response = await ws.recv()
        print(f"Server: {response}")

        print("Processing video...")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break

            # Send raw BGR bytes
            await ws.send(frame.tobytes())

            # Receive response
            response = await ws.recv()
            frame_count += 1
            print(f"Frame {frame_count}: {response}")

            # Display the frame locally
            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Match video frame rate
            await asyncio.sleep(frame_delay)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video.py <video_path>")
        sys.exit(1)

    asyncio.run(send_video(sys.argv[1]))
