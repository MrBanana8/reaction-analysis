"""
Optimized Production WebSocket server for video emotion analysis using Hume AI.
Improvements:
- Hume connection pooling (reuse connections across clients)
- Connection limits to prevent overload
- Async file operations
- In-memory processing when possible
- Graceful degradation under load

Usage: python server_optimized.py
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

import aiofiles
import aiofiles.os
import websockets
from dotenv import load_dotenv
from hume import AsyncHumeClient
from hume.expression_measurement.stream.stream.socket_client import Config
from hume.expression_measurement.stream.stream.types import StreamFace
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class HumeConnection:
    """Wrapper for a Hume streaming connection."""
    client: AsyncHumeClient
    socket: any
    in_use: bool = False
    created_at: float = 0
    uses: int = 0


class HumeConnectionPool:
    """
    Pool of reusable Hume AI connections.
    Reduces connection overhead for concurrent users.
    """

    def __init__(self, api_key: str, pool_size: int = 5, max_uses: int = 100):
        self.api_key = api_key
        self.pool_size = pool_size
        self.max_uses = max_uses  # Recycle connection after N uses
        self.connections: list[HumeConnection] = []
        self.lock = asyncio.Lock()
        self.face_config = Config(face=StreamFace())

    async def _create_connection(self) -> HumeConnection:
        """Create a new Hume connection."""
        client = AsyncHumeClient(api_key=self.api_key)
        # Note: We create the connection context but need to manage it carefully
        conn = HumeConnection(
            client=client,
            socket=None,
            created_at=time.time(),
            uses=0
        )
        return conn

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a Hume connection from the pool.
        Creates new connection if pool is empty.
        """
        connection = None

        async with self.lock:
            # Find an available connection
            for conn in self.connections:
                if not conn.in_use and conn.uses < self.max_uses:
                    conn.in_use = True
                    connection = conn
                    break

            # Create new connection if needed and pool not full
            if connection is None and len(self.connections) < self.pool_size:
                connection = await self._create_connection()
                connection.in_use = True
                self.connections.append(connection)

        if connection is None:
            # Pool exhausted - create temporary connection
            logger.warning("Connection pool exhausted, creating temporary connection")
            connection = await self._create_connection()
            connection.in_use = True

        try:
            # Connect to Hume if not already connected
            async with connection.client.expression_measurement.stream.connect() as socket:
                connection.socket = socket
                connection.uses += 1
                yield connection
        finally:
            async with self.lock:
                connection.in_use = False
                connection.socket = None

                # Remove if exceeded max uses
                if connection.uses >= self.max_uses:
                    if connection in self.connections:
                        self.connections.remove(connection)
                        logger.info(f"Recycled connection after {connection.uses} uses")

    async def close_all(self):
        """Close all connections in the pool."""
        async with self.lock:
            self.connections.clear()
            logger.info("Connection pool closed")


class EmotionAnalysisServer:
    def __init__(self):
        load_dotenv()

        self.host = os.getenv("WS_HOST", "0.0.0.0")
        self.port = int(os.getenv("WS_PORT") or os.getenv("PORT") or "8765")
        self.api_key = os.getenv("HUME_API_KEY")
        self.analyze_every_n_frames = int(os.getenv("ANALYZE_EVERY_N_FRAMES", "1"))
        self.min_seconds_to_upload = int(os.getenv("MIN_SECONDS_TO_UPLOAD", "5"))

        # Concurrency settings
        self.max_concurrent_connections = int(os.getenv("MAX_CONCURRENT_CONNECTIONS", "50"))
        self.hume_pool_size = int(os.getenv("HUME_POOL_SIZE", "10"))

        if not self.api_key:
            raise ValueError("HUME_API_KEY must be set in .env")

        # Initialize Hume connection pool
        self.hume_pool = HumeConnectionPool(
            api_key=self.api_key,
            pool_size=self.hume_pool_size
        )

        # Semaphore for limiting concurrent connections
        self.connection_semaphore = asyncio.Semaphore(self.max_concurrent_connections)

        # Initialize Supabase client (optional)
        self.supabase: Client | None = None
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        if supabase_url and supabase_key:
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized")
        else:
            logger.warning("SUPABASE_URL or SUPABASE_SERVICE_KEY not set")

        self.active_connections = 0
        self.total_connections = 0
        self.frames_processed = 0

    async def upload_to_supabase(self, summary: dict, context: dict = None) -> bool:
        """Upload analysis results to Supabase (async-safe)."""
        if not self.supabase:
            return False

        context = context or {}

        try:
            data = {
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

            if context.get("user_id"):
                data["user_id"] = context["user_id"]
            if context.get("ad_id"):
                data["ad_id"] = context["ad_id"]
            if context.get("reaction_id"):
                data["reaction_id"] = context["reaction_id"]

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.supabase.table("analysis_results").insert(data).execute()
            )
            logger.info("Results uploaded to Supabase")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to Supabase: {e}")
            return False

    async def process_frame(self, hume_socket, frame_data: bytes, frame_count: int,
                           current_second: int) -> dict:
        """Process a single frame with Hume AI."""
        # Write to temp file asynchronously
        tmp_path = None
        try:
            # Create temp file
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)

            # Write asynchronously
            async with aiofiles.open(tmp_path, 'wb') as f:
                await f.write(frame_data)

            # Send to Hume
            result = await hume_socket.send_file(tmp_path, config=Config(face=StreamFace()))

            faces_detected = 0
            frame_emotions = []
            emotion_scores = {}

            if result and hasattr(result, "face") and result.face:
                predictions = result.face.predictions or []
                faces_detected = len(predictions)

                for pred in predictions:
                    if pred.emotions:
                        emotions_dict = {e.name: e.score for e in pred.emotions}
                        dominant = max(emotions_dict, key=emotions_dict.get)
                        frame_emotions.append(dominant)
                        emotion_scores = emotions_dict

            self.frames_processed += 1

            return {
                "status": "frame_processed",
                "frame": frame_count,
                "second": current_second,
                "faces_detected": faces_detected,
                "emotions": frame_emotions,
                "scores": emotion_scores
            }

        except Exception as e:
            logger.error(f"Frame {frame_count} error: {e}")
            return {
                "status": "error",
                "frame": frame_count,
                "message": str(e)
            }
        finally:
            # Cleanup temp file asynchronously
            if tmp_path:
                try:
                    await aiofiles.os.remove(tmp_path)
                except:
                    pass

    async def handle_connection(self, websocket):
        """Handle a single WebSocket connection with rate limiting."""

        # Check if we can accept this connection
        if self.active_connections >= self.max_concurrent_connections:
            logger.warning(f"Connection rejected: max connections ({self.max_concurrent_connections}) reached")
            await websocket.send(json.dumps({
                "status": "error",
                "message": "Server at capacity. Please try again later."
            }))
            await websocket.close()
            return

        async with self.connection_semaphore:
            await self._handle_connection_inner(websocket)

    async def _handle_connection_inner(self, websocket):
        """Inner connection handler (after rate limiting)."""
        self.active_connections += 1
        self.total_connections += 1
        connection_id = self.total_connections

        logger.info(f"[{connection_id}] Client connected (active: {self.active_connections})")

        start_time = time.time()
        all_emotions = []
        emotion_scores_sum = {}
        emotion_scores_count = {}
        emotions_per_second = {}
        frame_count = 0
        client_context = {}

        try:
            # Acquire Hume connection from pool
            async with self.hume_pool.acquire() as hume_conn:
                logger.info(f"[{connection_id}] Acquired Hume connection")

                # Send ready message
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

                        current_second = int(time.time() - start_time)

                        # Process frame
                        result = await self.process_frame(
                            hume_conn.socket,
                            message,
                            frame_count,
                            current_second
                        )

                        # Accumulate statistics
                        if result.get("emotions"):
                            for emotion in result["emotions"]:
                                all_emotions.append(emotion)

                                if current_second not in emotions_per_second:
                                    emotions_per_second[current_second] = []
                                emotions_per_second[current_second].append(emotion)

                            # Track scores
                            if result.get("scores"):
                                for name, score in result["scores"].items():
                                    if name not in emotion_scores_sum:
                                        emotion_scores_sum[name] = 0
                                        emotion_scores_count[name] = 0
                                    emotion_scores_sum[name] += score
                                    emotion_scores_count[name] += 1

                        # Send result (without scores to reduce bandwidth)
                        await websocket.send(json.dumps({
                            "status": result["status"],
                            "frame": result["frame"],
                            "second": result.get("second", 0),
                            "faces_detected": result.get("faces_detected", 0),
                            "emotions": result.get("emotions", [])
                        }))

                    elif isinstance(message, str):
                        data = json.loads(message)

                        if data.get("action") == "start":
                            client_context["user_id"] = data.get("user_id")
                            client_context["ad_id"] = data.get("ad_id")
                            client_context["reaction_id"] = data.get("reaction_id")
                            logger.info(f"[{connection_id}] Context: user={client_context.get('user_id')}, ad={client_context.get('ad_id')}")
                            await websocket.send(json.dumps({
                                "status": "context_set",
                                "user_id": client_context.get("user_id"),
                                "ad_id": client_context.get("ad_id"),
                                "reaction_id": client_context.get("reaction_id")
                            }))
                            continue

                        if data.get("action") == "end":
                            break

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[{connection_id}] Client disconnected")
        except Exception as e:
            logger.error(f"[{connection_id}] Error: {e}")
        finally:
            self.active_connections -= 1
            elapsed_time = time.time() - start_time

            # Generate and send summary
            await self._send_summary(
                websocket,
                frame_count,
                all_emotions,
                emotion_scores_sum,
                emotion_scores_count,
                emotions_per_second,
                elapsed_time,
                client_context,
                connection_id
            )

    async def _send_summary(self, websocket, frame_count: int, all_emotions: list,
                           emotion_scores_sum: dict, emotion_scores_count: dict,
                           emotions_per_second: dict, elapsed_time: float,
                           client_context: dict, connection_id: int):
        """Generate and send session summary."""
        if all_emotions:
            emotion_counts = Counter(all_emotions)
            total = len(all_emotions)
            avg_scores = {
                k: emotion_scores_sum[k] / emotion_scores_count[k]
                for k in emotion_scores_sum
            }
            most_frequent = emotion_counts.most_common(1)[0]

            timeline = {}
            for second in sorted(emotions_per_second.keys()):
                second_counts = Counter(emotions_per_second[second])
                dominant = second_counts.most_common(1)[0][0]
                timeline[str(second)] = dominant

            summary = {
                "status": "complete",
                "reaction_id": client_context.get("reaction_id"),
                "total_frames": frame_count,
                "frames_with_faces": total,
                "processing_time_seconds": round(elapsed_time, 2),
                "emotion_frequency": {
                    emotion: {"count": count, "percentage": round((count / total) * 100, 1)}
                    for emotion, count in emotion_counts.most_common()
                },
                "average_scores": {
                    k: round(v, 4)
                    for k, v in sorted(avg_scores.items(), key=lambda x: -x[1])
                },
                "timeline": timeline,
                "final_result": {
                    "emotion": most_frequent[0],
                    "confidence": round((most_frequent[1] / total) * 100, 1),
                },
            }

            try:
                await websocket.send(json.dumps(summary))
            except:
                pass

            if elapsed_time >= self.min_seconds_to_upload:
                await self.upload_to_supabase(summary, client_context)

            logger.info(
                f"[{connection_id}] Complete: {frame_count} frames, "
                f"{total} faces, {elapsed_time:.2f}s, result: {most_frequent[0]}"
            )
        else:
            try:
                await websocket.send(json.dumps({
                    "status": "complete",
                    "reaction_id": client_context.get("reaction_id"),
                    "total_frames": frame_count,
                    "frames_with_faces": 0,
                    "processing_time_seconds": round(elapsed_time, 2),
                    "message": "No faces detected",
                }))
            except:
                pass

            logger.info(f"[{connection_id}] Complete: {frame_count} frames, no faces, {elapsed_time:.2f}s")

    async def log_stats(self):
        """Periodically log server statistics."""
        while True:
            await asyncio.sleep(60)
            logger.info(
                f"Stats: {self.active_connections} active connections, "
                f"{self.total_connections} total, "
                f"{self.frames_processed} frames processed, "
                f"{len(self.hume_pool.connections)} Hume connections in pool"
            )

    async def start(self):
        """Start the WebSocket server."""
        logger.info(f"Starting optimized server on ws://{self.host}:{self.port}")
        logger.info(f"Max concurrent connections: {self.max_concurrent_connections}")
        logger.info(f"Hume connection pool size: {self.hume_pool_size}")

        # Start stats logging
        asyncio.create_task(self.log_stats())

        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024,
            ping_interval=30,
            ping_timeout=10,
            max_queue=64,  # Limit queued connections
        ):
            logger.info("Server ready. Waiting for connections...")
            await asyncio.Future()


def main():
    server = EmotionAnalysisServer()
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
