import json
import sys
from collections import Counter


def summarize_results(file_path: str = "emotion_results.json"):
    with open(file_path) as f:
        results = json.load(f)

    if not results:
        print("No results to summarize")
        return

    # Count dominant emotions
    emotions = [r["dominant_emotion"] for r in results]
    emotion_counts = Counter(emotions)
    total = len(emotions)

    # Calculate average scores across all frames
    avg_scores = {}
    for r in results:
        for emotion, score in r["emotion_scores"].items():
            if emotion not in avg_scores:
                avg_scores[emotion] = []
            avg_scores[emotion].append(score)

    avg_scores = {k: sum(v) / len(v) for k, v in avg_scores.items()}

    # Find the most frequent emotion
    most_frequent = emotion_counts.most_common(1)[0]

    # Print summary
    print("=" * 50)
    print("EMOTION ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"\nTotal frames analyzed: {total}")
    print(f"\nEmotion frequency:")
    for emotion, count in emotion_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {emotion}: {count} ({percentage:.1f}%)")

    print(f"\nAverage emotion scores:")
    for emotion, score in sorted(avg_scores.items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {score:.2f}%")

    print("\n" + "=" * 50)
    print(f"FINAL RESULT: {most_frequent[0].upper()}")
    print(f"Confidence: {most_frequent[1]}/{total} frames ({(most_frequent[1]/total)*100:.1f}%)")
    print("=" * 50)

    return {
        "total_frames": total,
        "emotion_counts": dict(emotion_counts),
        "average_scores": avg_scores,
        "final_emotion": most_frequent[0],
        "confidence": most_frequent[1] / total,
    }


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "emotion_results.json"
    summarize_results(file_path)
