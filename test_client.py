import asyncio
import json
import cv2
import websockets


async def send_video():
    uri = "ws://localhost:8765"

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Get webcam dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Webcam: {frame_width}x{frame_height}")
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri, max_size=10 * 1024 * 1024) as ws:
        # Send dimensions first
        await ws.send(json.dumps({"width": frame_width, "height": frame_height}))
        response = await ws.recv()
        print(f"Server: {response}")

        print("Connected! Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Send raw BGR bytes
            await ws.send(frame.tobytes())

            # Receive response
            response = await ws.recv()
            print(f"Response: {response}")

            # Display the frame locally
            cv2.imshow("Sending", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Limit frame rate
            await asyncio.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(send_video())
