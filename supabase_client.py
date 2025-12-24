import os
from datetime import datetime, timezone

from supabase import create_client, Client


class SupabaseClient:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

        self.client: Client = create_client(url, key)
        self.table_name = "emotion_results"

    def store_emotion_result(
        self,
        dominant_emotion: str,
        emotion_scores: dict,
        session_id: str | None = None,
    ) -> dict | None:
        """
        Store an emotion analysis result in Supabase.

        Args:
            dominant_emotion: The dominant emotion detected
            emotion_scores: Dictionary of all emotion scores
            session_id: Optional session identifier

        Returns:
            The inserted record or None on error
        """
        try:
            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dominant_emotion": dominant_emotion,
                "emotion_scores": emotion_scores,
            }

            if session_id:
                data["session_id"] = session_id

            result = self.client.table(self.table_name).insert(data).execute()
            return result.data[0] if result.data else None

        except Exception as e:
            print(f"Error storing emotion result: {e}")
            return None

    def store_batch_results(
        self,
        results: list[dict],
        session_id: str | None = None,
    ) -> list[dict]:
        """
        Store multiple emotion results in a single batch.

        Args:
            results: List of emotion result dicts with dominant_emotion and emotion_scores
            session_id: Optional session identifier

        Returns:
            List of inserted records
        """
        if not results:
            return []

        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            data = [
                {
                    "timestamp": timestamp,
                    "dominant_emotion": r["dominant_emotion"],
                    "emotion_scores": r["emotion_scores"],
                    **({"session_id": session_id} if session_id else {}),
                }
                for r in results
            ]

            result = self.client.table(self.table_name).insert(data).execute()
            return result.data if result.data else []

        except Exception as e:
            print(f"Error storing batch results: {e}")
            return []
