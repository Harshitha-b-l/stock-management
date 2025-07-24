import streamlit as st
import pandas as pd
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import os

# --------------------------
# CATEGORY MAPPING DICTIONARY
# --------------------------
CATEGORY_MAP = {
    "bottle": "Beverages",
    "cup": "Beverages",
    "cell phone": "Electronics",
    "laptop": "Electronics",
    "tv": "Electronics",
    "book": "Stationery",
    "toothbrush": "Personal Care",
    "hair drier": "Personal Care",
    "apple": "Fruits",
    "orange": "Fruits",
    "banana": "Fruits",
    "phone": "Electronics",
    "person": "Human"  # fixed: 'human' → 'person' as per COCO
}

# --------------------------
# LOAD MODEL
# --------------------------
model = YOLO("yolov8n.pt")  # uses COCO-pretrained classes

# --------------------------
# CSV LOG PATH
# --------------------------
csv_path = "stock_log.csv"

# --------------------------
# STREAMLIT UI
# --------------------------
st.set_page_config(page_title="Stock Detection", layout="centered")
st.title("📦 Stock Detection with YOLOv8 + Streamlit")
st.markdown("Take a photo using webcam to detect items and categorize them automatically.")

# --------------------------
# CAMERA INPUT
# --------------------------
image = st.camera_input("📷 Take a Photo of Your Stock")

if image is not None:
    # Decode image to numpy
    img_array = np.frombuffer(image.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Show annotated image
    st.image(annotated_frame, caption="🧠 YOLOv8 Detection", use_column_width=True)

    # Get detections
    names = results[0].names
    classes = results[0].boxes.cls

    st.subheader("📋 Detected Items")
    log_data = []

    if len(classes) == 0:
        st.warning("No objects detected.")
    else:
        for cls in classes:
            product = names[int(cls)]
            category = CATEGORY_MAP.get(product, "Uncategorized")
            timestamp = datetime.now()

            log_data.append({
                "product_name": product,
                "category": category,
                "timestamp": timestamp,
                "quantity": 1
            })
            st.write(f"✅ {product} → {category}")

        # Save to CSV
        df = pd.DataFrame(log_data)
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)
        st.success(f"{len(log_data)} item(s) logged to *stock_log.csv* ✅")

# --------------------------
# SHOW HISTORY + CATEGORY SUMMARY
# --------------------------
# --------------------------
# SHOW HISTORY + CATEGORY SUMMARY (FULLY SAFE)
# --------------------------
if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
    try:
        df_log = pd.read_csv(csv_path)

        st.markdown("### 🕓 Detection History")
        st.dataframe(df_log.tail(10))

        st.markdown("### 📊 Category-wise Summary")
        category_counts = df_log["category"].value_counts().reset_index()
        category_counts.columns = ["Category", "Count"]
        st.dataframe(category_counts)
        st.bar_chart(category_counts.set_index("Category"))

    except Exception as e:
        st.error(f"⚠ Failed to read stock_log.csv: {e}")
        st.info("Try deleting the CSV and re-running the detection.")
else:
    st.info("No detection data available yet. Please take a photo to start logging.")