import asyncio
import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone

import websockets
from websockets.server import WebSocketServerProtocol

from emotion_analyzer import EmotionAnalyzer


class VideoEmotionServer:
    def __init__(self):
        self.host = os.getenv("WS_HOST", "0.0.0.0")
        self.port = int(os.getenv("WS_PORT", "8765"))
        self.output_file = os.getenv("OUTPUT_FILE", "emotion_results.json")
        self.analyze_every_n = int(os.getenv("ANALYZE_EVERY_N_FRAMES", "1"))

        self.analyzer = EmotionAnalyzer()
        self.results = []
        self.frame_count = 0
        self.frame_width = None
        self.frame_height = None

    def summarize_and_save(self):
        """Summarize results and save to JSON file."""
        if not self.results:
            print("No results to summarize")
            return

        # Count dominant emotions
        emotions = [r["dominant_emotion"] for r in self.results]
        emotion_counts = Counter(emotions)
        total = len(emotions)

        # Calculate average scores across all frames
        avg_scores = {}
        for r in self.results:
            for emotion, score in r["emotion_scores"].items():
                if emotion not in avg_scores:
                    avg_scores[emotion] = []
                avg_scores[emotion].append(score)

        avg_scores = {k: sum(v) / len(v) for k, v in avg_scores.items()}

        # Find the most frequent emotion
        most_frequent = emotion_counts.most_common(1)[0]

        # Build summary
        summary = {
            "total_frames_analyzed": total,
            "emotion_frequency": {
                emotion: {"count": count, "percentage": (count / total) * 100}
                for emotion, count in emotion_counts.most_common()
            },
            "average_scores": avg_scores,
            "final_result": {
                "emotion": most_frequent[0],
                "confidence": (most_frequent[1] / total) * 100,
            },
        }

        # Save to file
        with open(self.output_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "=" * 50)
        print("EMOTION ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"\nTotal frames analyzed: {total}")
        print(f"\nEmotion frequency:")
        for emotion, count in emotion_counts.most_common():
            percentage = (count / total) * 100
            print(f"  {emotion}: {count} ({percentage:.1f}%)")
        print("\n" + "=" * 50)
        print(f"FINAL RESULT: {most_frequent[0].upper()}")
        print(f"Confidence: {most_frequent[1]}/{total} frames ({(most_frequent[1]/total)*100:.1f}%)")
        print("=" * 50 + "\n")

    async def handle_connection(self, websocket: WebSocketServerProtocol):
        """Handle a single WebSocket connection."""
        session_id = str(uuid.uuid4())
        print(f"New connection: {session_id}")
        self.frame_count = 0
        self.frame_width = None
        self.frame_height = None
        self.results = []

        try:
            async for message in websocket:
                # First message should be JSON with dimensions
                if self.frame_width is None:
                    if isinstance(message, str):
                        try:
                            config = json.loads(message)
                            self.frame_width = config["width"]
                            self.frame_height = config["height"]
                            print(f"Frame size set: {self.frame_width}x{self.frame_height}")
                            await websocket.send(json.dumps({"status": "ready"}))
                        except (json.JSONDecodeError, KeyError) as e:
                            await websocket.send(json.dumps({
                                "status": "error",
                                "message": "First message must be JSON with 'width' and 'height'",
                            }))
                    else:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": "First message must be JSON with 'width' and 'height'",
                        }))
                elif isinstance(message, bytes):
                    await self.process_frame(message, session_id, websocket)

        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed: {session_id}")
        except Exception as e:
            print(f"Error in connection {session_id}: {e}")
        finally:
            self.summarize_and_save()
            print(f"Session ended: {session_id}")
            print(f"Summary saved to {self.output_file}")

    async def process_frame(
        self,
        data: bytes,
        session_id: str,
        websocket: WebSocketServerProtocol,
    ):
        """Process a single frame: analyze emotions and save results."""
        self.frame_count += 1

        # Skip frames based on ANALYZE_EVERY_N_FRAMES setting
        if self.frame_count % self.analyze_every_n != 0:
            await websocket.send(json.dumps({
                "status": "skipped",
                "frame": self.frame_count,
            }))
            return

        try:
            frame = self.analyzer.bytes_to_frame(
                data, self.frame_width, self.frame_height
            )

            results = await asyncio.get_event_loop().run_in_executor(
                None, self.analyzer.analyze_frame, frame
            )

            if results:
                for r in results:
                    self.results.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "session_id": session_id,
                        "frame": self.frame_count,
                        "dominant_emotion": r["dominant_emotion"],
                        "emotion_scores": r["emotion_scores"],
                    })

                response = json.dumps({
                    "status": "success",
                    "faces_detected": len(results),
                    "emotions": [r["dominant_emotion"] for r in results],
                })
                await websocket.send(response)
            else:
                await websocket.send(json.dumps({
                    "status": "success",
                    "faces_detected": 0,
                    "emotions": [],
                }))

        except ValueError as e:
            await websocket.send(json.dumps({
                "status": "error",
                "message": str(e),
            }))
        except Exception as e:
            print(f"Error processing frame: {e}")
            await websocket.send(json.dumps({
                "status": "error",
                "message": "Internal processing error",
            }))

    async def start(self):
        """Start the WebSocket server."""
        print(f"Starting server on ws://{self.host}:{self.port}")
        print(f"Analyzing every {self.analyze_every_n} frame(s)")
        print(f"Results will be saved to: {self.output_file}")
        print("Waiting for client to send frame dimensions...")

        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024,  # 10MB max to handle various resolutions
        ):
            await asyncio.Future()
