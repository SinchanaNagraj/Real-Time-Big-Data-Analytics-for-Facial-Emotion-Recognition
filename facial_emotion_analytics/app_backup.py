import os
import time
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from deepface import DeepFace

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "emotion_events.db"
CSV_PATH = OUTPUT_DIR / "emotion_events.csv"

st.set_page_config(page_title="Real-Time Facial Emotion Analytics", page_icon="🙂", layout="wide")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                face_count INTEGER NOT NULL
            )
        """)

def log_event(source, emotion, confidence, face_count=1):
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO events(timestamp, source, emotion, confidence, face_count) VALUES (?, ?, ?, ?, ?)",
            (now, source, emotion, float(confidence), int(face_count))
        )

def load_events():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM events ORDER BY timestamp DESC", conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def analyze_image(pil_image, source):
    rgb = np.array(pil_image.convert("RGB"))

    results = DeepFace.analyze(
        img_path=rgb,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="retinaface",
        silent=True
    )

    if isinstance(results, dict):
        results = [results]

    candidates = []

    # Collect emotion predictions and face regions.
    for item in results:
        emotion_scores = item.get("emotion", {}) or {}

        if not emotion_scores:
            continue

        emotion = str(
            item.get(
                "dominant_emotion",
                max(emotion_scores, key=emotion_scores.get)
            )
        ).lower()

        confidence = float(
            emotion_scores.get(
                emotion,
                max(emotion_scores.values(), default=0.0)
            )
        )

        region = item.get("region") or {}

        try:
            x = float(region["x"])
            y = float(region["y"])
            w = float(region["w"])
            h = float(region["h"])

            # Ignore invalid or empty face regions.
            if w <= 0 or h <= 0:
                region = None
        except (KeyError, TypeError, ValueError):
            region = None

        try:
            face_score = float(item.get("face_confidence", 0.0))
        except (TypeError, ValueError):
            face_score = 0.0

        candidates.append({
            "emotion": emotion,
            "confidence": confidence,
            "region": region,
            "face_score": face_score
        })

    # Compare two face boxes to determine how much they overlap.
    def calculate_iou(region_a, region_b):
        ax1 = region_a["x"]
        ay1 = region_a["y"]
        ax2 = ax1 + region_a["w"]
        ay2 = ay1 + region_a["h"]

        bx1 = region_b["x"]
        by1 = region_b["y"]
        bx2 = bx1 + region_b["w"]
        by2 = by1 + region_b["h"]

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection = (
            max(0.0, ix2 - ix1) *
            max(0.0, iy2 - iy1)
        )

        area_a = region_a["w"] * region_a["h"]
        area_b = region_b["w"] * region_b["h"]
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    # Prioritize stronger face detections.
    candidates.sort(
        key=lambda face: face["face_score"],
        reverse=True
    )

    unique_faces = []

    for candidate in candidates:
        duplicate = False

        if candidate["region"] is not None:
            for kept in unique_faces:
                if kept["region"] is None:
                    continue

                overlap = calculate_iou(
                    candidate["region"],
                    kept["region"]
                )

                # Suppress highly overlapping face detections.
                if overlap >= 0.65:
                    duplicate = True
                    break

        if not duplicate:
            unique_faces.append(candidate)

    # Preserve the existing output format for the UI.
    valid = [
        (face["emotion"], face["confidence"])
        for face in unique_faces
    ]

    # Store only the retained predictions.
    for emotion, confidence in valid:
        log_event(source, emotion, confidence, len(valid))

    return valid



def analyze_video(video_bytes, filename, frame_step=30, max_frames=40):
    suffix = Path(filename).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(video_bytes)
        temp_path = temp.name

    records = []
    history = []
    cap = cv2.VideoCapture(temp_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    frame_idx = 0
    processed = 0
    progress = st.progress(0, text="Preparing video...")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    try:
        while cap.isOpened() and processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    result = DeepFace.analyze(
                        img_path=rgb,
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend="opencv",
                        silent=True
                    )

                    if isinstance(result, dict):
                        result = [result]

                    frame_predictions = []

                    for item in result:
                        scores = item.get("emotion", {}) or {}
                        if not scores:
                            continue

                        raw_emotion = str(
                            item.get(
                                "dominant_emotion",
                                max(scores, key=scores.get)
                            )
                        ).lower()

                        frame_predictions.append({
                            "raw_emotion": raw_emotion,
                            "scores": {
                                str(k).lower(): float(v)
                                for k, v in scores.items()
                            }
                        })

                    # Smooth predictions over the latest five
                    # sampled frames. Intended for a single face.
                    if frame_predictions:
                        history.extend(frame_predictions)
                        history = history[-5:]

                        labels = [
                            entry["raw_emotion"]
                            for entry in history
                        ]

                        smoothed_emotion = max(
                            set(labels),
                            key=labels.count
                        )

                        matching_scores = [
                            entry["scores"][smoothed_emotion]
                            for entry in history
                            if smoothed_emotion in entry["scores"]
                        ]

                        smoothed_confidence = (
                            sum(matching_scores) / len(matching_scores)
                            if matching_scores else 0.0
                        )

                        seconds = frame_idx / fps
                        timestamp = datetime.now().isoformat(
                            timespec="seconds"
                        )

                        records.append({
                            "timestamp": timestamp,
                            "source": f"video:{filename}",
                            "emotion": smoothed_emotion,
                            "raw_emotion": frame_predictions[0][
                                "raw_emotion"
                            ],
                            "confidence": round(
                                smoothed_confidence, 2
                            ),
                            "face_count": len(frame_predictions),
                            "video_second": round(seconds, 2),
                            "frame": frame_idx
                        })

                        log_event(
                            f"video:{filename}",
                            smoothed_emotion,
                            smoothed_confidence,
                            len(frame_predictions)
                        )

                except Exception as exc:
                    print(
                        f"Video frame {frame_idx} failed: {exc}"
                    )

                processed += 1

                if total:
                    progress.progress(
                        min(frame_idx / total, 1.0),
                        text=f"Processed {processed} sampled frames..."
                    )

            frame_idx += 1

    finally:
        cap.release()
        progress.empty()

        try:
            os.remove(temp_path)
        except OSError:
            pass

    return pd.DataFrame(records)


init_db()
st.title("🙂 Real-Time Big Data Analytics for Facial Emotion Recognition")
st.caption("Facial-expression predictions from images, webcam snapshots, and sampled video frames, with timestamped analytics.")
st.info("Use only images/videos you have permission to process. Facial expressions are imperfect signals and do not reliably reveal a person's true emotional state.")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Choose input mode", ["Upload image", "Webcam snapshot", "Upload video", "Analytics dashboard"])
    st.divider()
    st.caption("Model: pretrained DeepFace emotion analyzer")
    st.caption("Storage: SQLite + CSV export")
    st.caption("Analytics: Pandas + Plotly")

if mode == "Upload image":
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input image", width=420)
        if st.button("Analyze image", type="primary"):
            with st.spinner("Loading model / analyzing image (first run can take longer)..."):
                try:
                    predictions = analyze_image(image, f"image:{uploaded.name}")
                    if predictions:
                        cols = st.columns(min(len(predictions), 4))
                        for i, (emotion, confidence) in enumerate(predictions):
                            with cols[i % len(cols)]:
                                st.metric(f"Face {i+1}: {emotion.title()}", f"{confidence:.1f}%")
                        st.success(f"Processed image. Faces with emotion output: {len(predictions)}")
                    else:
                        st.warning("No usable face/emotion output was detected. Try a clear, front-facing image with good lighting.")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.caption("If this is the first run, check your internet connection so model weights can download.")

elif mode == "Webcam snapshot":
    camera = st.camera_input("Take a snapshot")
    if camera:
        image = Image.open(camera).convert("RGB")
        if st.button("Analyze snapshot", type="primary"):
            with st.spinner("Analyzing snapshot..."):
                try:
                    predictions = analyze_image(image, "webcam_snapshot")
                    if predictions:
                        for i, (emotion, confidence) in enumerate(predictions, 1):
                            st.metric(f"Face {i}: {emotion.title()}", f"{confidence:.1f}%")
                    else:
                        st.warning("No usable face/emotion output detected. Try again with better lighting.")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

elif mode == "Upload video":
    st.write("Upload a short video. The app samples frames to keep processing time manageable.")
    video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    frame_step = st.slider("Analyze every Nth frame", min_value=5, max_value=90, value=30, step=5)
    max_frames = st.slider("Maximum sampled frames", min_value=5, max_value=100, value=30, step=5)
    if video and st.button("Process video", type="primary"):
        with st.spinner("Processing sampled frames. This can take a few minutes..."):
            try:
                video_df = analyze_video(video.getvalue(), video.name, frame_step, max_frames)
                if video_df.empty:
                    st.warning("No emotion outputs were produced. Try another short video with visible faces.")
                else:
                    st.success(f"Processed {len(video_df)} face-level emotion outputs.")

                    # -------- VIDEO EMOTION SUMMARY --------
                    st.subheader("Video Emotion Summary")

                    emotion_counts = (
                        video_df["emotion"]
                        .astype(str)
                        .str.lower()
                        .value_counts()
                    )

                    total_predictions = len(video_df)
                    top_emotion = emotion_counts.index[0]
                    top_percentage = (
                        emotion_counts.iloc[0] / total_predictions
                    ) * 100

                    if "frame" in video_df.columns:
                        frames_analyzed = video_df["frame"].nunique()
                    else:
                        frames_analyzed = total_predictions

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Frames Analyzed", frames_analyzed)
                    col2.metric("Face-Level Predictions", total_predictions)
                    col3.metric(
                        "Most Frequent Emotion",
                        top_emotion.title(),
                        f"{top_percentage:.1f}% of predictions"
                    )

                    summary_df = emotion_counts.rename_axis(
                        "Emotion"
                    ).reset_index(name="Count")
                    summary_df["Percentage"] = (
                        summary_df["Count"] / total_predictions * 100
                    ).round(1)
                    summary_df["Emotion"] = summary_df["Emotion"].str.title()

                    st.write("**Emotion Distribution**")
                    st.dataframe(summary_df, use_container_width=True)

                    # -------- KEEP EVERY FRAME'S PREDICTION --------
                    st.subheader("Frame-by-Frame Emotion Predictions")
                    st.caption(
                        "Each row preserves an individual prediction. "
                        "Emotion labels may change between sampled frames."
                    )
                    st.dataframe(video_df, use_container_width=True)
                    st.download_button(
                        "Download this video's results (CSV)",
                        video_df.to_csv(index=False).encode("utf-8"),
                        "video_emotion_results.csv",
                        "text/csv"
                    )

                    counts = (
                        video_df["emotion"]
                        .value_counts()
                        .rename_axis("emotion")
                        .reset_index(name="count")
                    )
                    st.plotly_chart(
                        px.bar(
                            counts,
                            x="emotion",
                            y="count",
                            title="Emotion Counts in Sampled Video Frames"
                        ),
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Video processing failed: {e}")
                st.caption("Try a short MP4 file. Some codecs may not be supported by OpenCV.")

else:
    st.subheader("Emotion analytics dashboard")
    df = load_events()
    if df.empty:
        st.warning("No records yet. Analyze an image, webcam snapshot, or video first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total prediction events", len(df))
        c2.metric("Unique categories", df["emotion"].nunique())
        c3.metric("Most frequent label", df["emotion"].mode().iloc[0].title())
        c4.metric("Mean model score", f"{df['confidence'].mean():.1f}%")
        left, right = st.columns(2)
        counts = df["emotion"].value_counts().rename_axis("emotion").reset_index(name="count")
        with left:
            st.plotly_chart(px.bar(counts, x="emotion", y="count", title="Emotion label distribution"), use_container_width=True)
        with right:
            st.plotly_chart(px.pie(counts, names="emotion", values="count", title="Share of recorded labels"), use_container_width=True)
        trend = df.dropna(subset=["timestamp"]).copy()
        if not trend.empty:
            trend["minute"] = trend["timestamp"].dt.floor("min")
            trend_counts = trend.groupby(["minute", "emotion"]).size().reset_index(name="count")
            st.plotly_chart(px.line(trend_counts, x="minute", y="count", color="emotion", markers=True, title="Recorded labels over time"), use_container_width=True)
        st.subheader("Stored prediction records")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download all records (CSV)", df.to_csv(index=False).encode("utf-8"), "emotion_events.csv", "text/csv")
        if st.button("Clear all stored records", type="secondary"):
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM events")
            st.success("Records cleared. Refresh the page to update the dashboard.")

st.divider()
st.caption("Prototype note: this is an analytics demonstration, not a validated psychological or clinical emotion measurement system.")
