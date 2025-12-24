import cv2
import numpy as np
from deepface import DeepFace


class EmotionAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def analyze_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Analyze emotions in a frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of emotion results for each detected face, each containing:
            - dominant_emotion: str
            - emotion_scores: dict of emotion -> confidence
        """
        results = []

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale to RGB for DeepFace (as per original implementation)
        rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for x, y, w, h in faces:
            # Extract face ROI from the RGB frame (not BGR)
            face_roi = rgb_frame[y : y + h, x : x + w]

            try:
                analysis = DeepFace.analyze(
                    face_roi,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )

                if isinstance(analysis, list):
                    analysis = analysis[0]

                # Convert numpy float32 to native Python floats for JSON serialization
                emotion_scores = {k: float(v) for k, v in analysis["emotion"].items()}

                results.append({
                    "dominant_emotion": analysis["dominant_emotion"],
                    "emotion_scores": emotion_scores,
                })
            except Exception as e:
                print(f"Error analyzing face: {e}")
                continue

        return results

    def bytes_to_frame(self, data: bytes, width: int, height: int) -> np.ndarray:
        """
        Convert raw bytes to a BGR frame.

        Args:
            data: Raw pixel bytes (BGR format, 3 channels)
            width: Frame width
            height: Frame height

        Returns:
            BGR image as numpy array
        """
        expected_size = width * height * 3
        if len(data) != expected_size:
            raise ValueError(
                f"Invalid frame size: expected {expected_size} bytes, got {len(data)}"
            )

        frame = np.frombuffer(data, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        return frame
