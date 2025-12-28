"""
Production WebSocket server for video emotion analysis using Hume AI.
Usage: python server.py
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from collections import Counter

import websockets
from dotenv import load_dotenv
from hume import AsyncHumeClient
from supabase import create_client, Client
from hume.expression_measurement.stream.stream.socket_client import Config
from hume.expression_measurement.stream.stream.types import StreamFace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EmotionAnalysisServer:
    def __init__(self):
        load_dotenv()

        self.host = os.getenv("WS_HOST", "0.0.0.0")
        self.port = int(os.getenv("WS_PORT", "8765"))
        self.api_key = os.getenv("HUME_API_KEY")
        self.analyze_every_n_frames = int(os.getenv("ANALYZE_EVERY_N_FRAMES", "1"))

        if not self.api_key:
            raise ValueError("HUME_API_KEY must be set in .env")

        # Initialize Supabase client (optional)
        self.supabase: Client | None = None
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        if supabase_url and supabase_key:
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized")
        else:
            logger.warning("SUPABASE_URL or SUPABASE_SERVICE_KEY not set - results will not be saved to database")

        self.active_connections = 0

    async def upload_to_supabase(self, summary: dict, context: dict = None) -> bool:
        """Upload analysis results to Supabase."""
        if not self.supabase:
            return False

        context = context or {}

        try:
            data = {
                "session_id": summary.get("session_id"),
                "total_frames": summary.get("total_frames"),
                "frames_with_faces": summary.get("frames_with_faces"),
                "processing_time_seconds": summary.get("processing_time_seconds"),
                "dominant_emotion": summary.get("final_result", {}).get("emotion"),
                "dominant_emotion_confidence": summary.get("final_result", {}).get("confidence"),
                "emotion_frequency": summary.get("emotion_frequency"),
                "average_scores": summary.get("average_scores"),
                "timeline": summary.get("timeline"),
                "raw_response": summary,
            }

            # Add context if provided
            if context.get("user_id"):
                data["user_id"] = context["user_id"]
            if context.get("ad_id"):
                data["ad_id"] = context["ad_id"]

            self.supabase.table("analysis_results").insert(data).execute()
            logger.info(f"[{summary.get('session_id')}] Results uploaded to Supabase")
            return True
        except Exception as e:
            logger.error(f"[{summary.get('session_id')}] Failed to upload to Supabase: {e}")
            return False

    async def handle_connection(self, websocket):
        """Handle a single WebSocket connection."""
        session_id = str(uuid.uuid4())[:8]
        self.active_connections += 1
        logger.info(f"[{session_id}] Client connected (active: {self.active_connections})")

        start_time = time.time()
        all_emotions = []
        emotion_scores_sum = {}
        emotion_scores_count = {}
        emotions_per_second = {}  # {second: [emotions]}
        frame_count = 0

        # Context from client (user_id, ad_id)
        client_context = {}

        try:
            # Connect to Hume streaming API
            client = AsyncHumeClient(api_key=self.api_key)
            face_config = Config(face=StreamFace())

            async with client.expression_measurement.stream.connect() as hume_socket:
                logger.info(f"[{session_id}] Connected to Hume AI")

                # Send ready message to client
                await websocket.send(json.dumps({"status": "ready"}))

                async for message in websocket:
                    if isinstance(message, bytes):
                        frame_count += 1

                        # Skip frames for performance
                        if frame_count % self.analyze_every_n_frames != 0:
                            await websocket.send(json.dumps({
                                "status": "frame_skipped",
                                "frame": frame_count,
                            }))
                            continue

                        # Save frame to temp file
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            tmp.write(message)
                            tmp_path = tmp.name

                        try:
                            # Send frame to Hume
                            result = await hume_socket.send_file(tmp_path, config=face_config)

                            # Calculate current second
                            current_second = int(time.time() - start_time)

                            faces_detected = 0
                            frame_emotions = []

                            # Process results
                            if result and hasattr(result, "face") and result.face:
                                predictions = result.face.predictions or []
                                faces_detected = len(predictions)

                                for pred in predictions:
                                    if pred.emotions:
                                        emotions_dict = {e.name: e.score for e in pred.emotions}
                                        dominant = max(emotions_dict, key=emotions_dict.get)
                                        all_emotions.append(dominant)
                                        frame_emotions.append(dominant)

                                        # Track emotion for this second
                                        if current_second not in emotions_per_second:
                                            emotions_per_second[current_second] = []
                                        emotions_per_second[current_second].append(dominant)

                                        for name, score in emotions_dict.items():
                                            if name not in emotion_scores_sum:
                                                emotion_scores_sum[name] = 0
                                                emotion_scores_count[name] = 0
                                            emotion_scores_sum[name] += score
                                            emotion_scores_count[name] += 1

                            # Send frame result to client
                            await websocket.send(json.dumps({
                                "status": "frame_processed",
                                "frame": frame_count,
                                "second": current_second,
                                "faces_detected": faces_detected,
                                "emotions": frame_emotions,
                            }))

                        except Exception as e:
                            logger.error(f"[{session_id}] Frame {frame_count} error: {e}")
                            await websocket.send(json.dumps({
                                "status": "error",
                                "frame": frame_count,
                                "message": str(e),
                            }))
                        finally:
                            os.unlink(tmp_path)

                    elif isinstance(message, str):
                        data = json.loads(message)

                        # Handle start action with context
                        if data.get("action") == "start":
                            client_context["user_id"] = data.get("user_id")
                            client_context["ad_id"] = data.get("ad_id")
                            logger.info(f"[{session_id}] Context set - user_id: {client_context.get('user_id')}, ad_id: {client_context.get('ad_id')}")
                            await websocket.send(json.dumps({"status": "context_set", "user_id": client_context.get("user_id"), "ad_id": client_context.get("ad_id")}))
                            continue

                        # Handle end of stream
                        if data.get("action") == "end":
                            break

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[{session_id}] Client disconnected")
        except Exception as e:
            logger.error(f"[{session_id}] Error: {e}")
        finally:
            self.active_connections -= 1
            elapsed_time = time.time() - start_time

            # Generate summary
            if all_emotions:
                emotion_counts = Counter(all_emotions)
                total = len(all_emotions)
                avg_scores = {k: emotion_scores_sum[k] / emotion_scores_count[k] for k in emotion_scores_sum}
                most_frequent = emotion_counts.most_common(1)[0]

                # Calculate dominant emotion per second
                timeline = {}
                for second in sorted(emotions_per_second.keys()):
                    second_counts = Counter(emotions_per_second[second])
                    dominant = second_counts.most_common(1)[0][0]
                    timeline[second] = dominant

                summary = {
                    "status": "complete",
                    "session_id": session_id,
                    "total_frames": frame_count,
                    "frames_with_faces": total,
                    "processing_time_seconds": round(elapsed_time, 2),
                    "emotion_frequency": {
                        emotion: {"count": count, "percentage": round((count / total) * 100, 1)}
                        for emotion, count in emotion_counts.most_common()
                    },
                    "average_scores": {k: round(v, 4) for k, v in sorted(avg_scores.items(), key=lambda x: -x[1])},
                    "timeline": timeline,  # Emotion per second
                    "final_result": {
                        "emotion": most_frequent[0],
                        "confidence": round((most_frequent[1] / total) * 100, 1),
                    },
                }

                try:
                    await websocket.send(json.dumps(summary))
                except:
                    pass

                # Upload to Supabase with context
                await self.upload_to_supabase(summary, client_context)

                logger.info(
                    f"[{session_id}] Complete: {frame_count} frames, "
                    f"{total} faces, {elapsed_time:.2f}s, result: {most_frequent[0]}"
                )
            else:
                try:
                    await websocket.send(json.dumps({
                        "status": "complete",
                        "session_id": session_id,
                        "total_frames": frame_count,
                        "frames_with_faces": 0,
                        "processing_time_seconds": round(elapsed_time, 2),
                        "message": "No faces detected",
                    }))
                except:
                    pass

                logger.info(f"[{session_id}] Complete: {frame_count} frames, no faces, {elapsed_time:.2f}s")

    async def start(self):
        """Start the WebSocket server."""
        logger.info(f"Starting server on ws://{self.host}:{self.port}")

        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024,  # 10MB max message
            ping_interval=30,
            ping_timeout=10,
        ):
            logger.info("Server ready. Waiting for connections...")
            await asyncio.Future()  # Run forever


def main():
    server = EmotionAnalysisServer()
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
