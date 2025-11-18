"""
extract_animal_features.py
-----------------------------------------
Extracts geometric facial features from YOLO-format bounding box labels
for animal images. Outputs a structured CSV with key metrics such as:

- EFR (Eye-to-Face Ratio)
- ESI (Eye Shape Index)
- Eye Symmetry Difference
- fWHR (Facial Width-to-Height Ratio)
- Ear Uprightness (left, right, average)

Author: Lizzie Qing
Project: Animal Facial Feature Dataset Construction
"""

import os
import numpy as np
import pandas as pd


# --------------------------------------------
# 1) YOLO bounding box → object name mapping
# --------------------------------------------
CLASS_ID_MAP = {
    0: "left_eye",
    1: "right_eye",
    2: "left_ear",
    3: "right_ear",
    4: "face"
}


# --------------------------------------------
# 2) Helper functions
# --------------------------------------------
def calc_esi(obj: dict) -> float:
    """
    Eye Shape Index (ESI): longer_side / shorter_side.
    """
    w, h = obj["width"], obj["height"]
    return max(w, h) / min(w, h) if min(w, h) != 0 else np.nan


def calc_uprightness(ear_obj: dict, face_obj: dict) -> float:
    """
    Computes ear uprightness angle in degrees.
    Smaller angle → more vertical ear.
    """
    dx = ear_obj["x_center"] - face_obj["x_center"]
    dy = face_obj["y_center"] - ear_obj["y_center"]

    if dy == 0:
        return 90.0  # perfectly horizontal
    return np.degrees(np.arctan2(abs(dx), abs(dy)))


def parse_yolo_label_file(filepath: str) -> dict:
    """
    Parses one YOLO .txt label file and returns a dict containing objects.
    """
    objects = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls_id = int(parts[0])

        if cls_id not in CLASS_ID_MAP:
            continue

        x_center, y_center, w, h = map(float, parts[1:])
        obj_name = CLASS_ID_MAP[cls_id]

        objects[obj_name] = {
            "x_center": x_center,
            "y_center": y_center,
            "width": w,
            "height": h
        }

    return objects


# --------------------------------------------
# 3) Main feature extraction logic
# --------------------------------------------
def extract_features_from_labels(labels_dir: str, output_csv: str):
    """
    Reads YOLO annotation files and computes all facial features.
    Saves to CSV.
    """
    results = []

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        filepath = os.path.join(labels_dir, label_file)
        filename = os.path.splitext(label_file)[0]

        data = parse_yolo_label_file(filepath)

        required = ["left_eye", "right_eye", "left_ear", "right_ear", "face"]
        if not all(k in data for k in required):
            print(f"⚠️ Missing key landmarks, skipped: {filename}")
            continue

        # ----- compute individual area metrics -----
        face_area = data["face"]["width"] * data["face"]["height"]
        left_eye_area = data["left_eye"]["width"] * data["left_eye"]["height"]
        right_eye_area = data["right_eye"]["width"] * data["right_eye"]["height"]

        # EFR
        efr = (left_eye_area + right_eye_area) / face_area if face_area != 0 else np.nan

        # ESI (avg between left & right)
        esi = np.nanmean([
            calc_esi(data["left_eye"]),
            calc_esi(data["right_eye"])
        ])

        # eye symmetry difference (percentage)
        eye_symmetry = (
            abs(left_eye_area - right_eye_area)
            / max(left_eye_area, right_eye_area)
            * 100
            if max(left_eye_area, right_eye_area) != 0 else np.nan
        )

        # fWHR
        fWHR = (
            data["face"]["width"] / data["face"]["height"]
            if data["face"]["height"] != 0 else np.nan
        )

        # Ear uprightness (both ears)
        left_up = calc_uprightness(data["left_ear"], data["face"])
        right_up = calc_uprightness(data["right_ear"], data["face"])
        avg_up = np.nanmean([left_up, right_up])

        results.append({
            "filename": filename,
            "left_eye_area": left_eye_area,
            "right_eye_area": right_eye_area,
            "eye_symmetry_diff_percent": eye_symmetry,
            "EFR": efr,
            "ESI": esi,
            "left_ear_uprightness_deg": left_up,
            "right_ear_uprightness_deg": right_up,
            "avg_ear_uprightness_deg": avg_up,
            "fWHR": fWHR
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Feature extraction completed! Saved to {output_csv}")


# --------------------------------------------
# 4) Script entry point
# --------------------------------------------
if __name__ == "__main__":
    labels_dir = "/Users/lizzie/PycharmProjects/pythonProject8/labels"
    output_csv = "animal_features_extracted.csv"

    extract_features_from_labels(labels_dir, output_csv)
