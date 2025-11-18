"""
batch_inference.py
---------------------------------------------------------
Runs batch inference using a trained YOLOv8 model.
Applies custom confidence thresholds for different classes
and saves annotated outputs.

Author: Lizzie Qing
"""

import os
import cv2
from ultralytics import YOLO


def run_batch_inference(
    model_path: str,
    input_folder: str,
    output_folder: str,
    face_conf_thresh: float = 0.45,
    other_conf_thresh: float = 0.25,
    imgsz: int = 640
):
    """
    Perform batch inference on all images inside a folder.

    Parameters
    ----------
    model_path : str
        Path to the trained YOLO model (best.pt).
    input_folder : str
        Folder containing images for inference.
    output_folder : str
        Output folder where annotated images will be saved.
    face_conf_thresh : float
        Minimum confidence required for class "face".
    other_conf_thresh : float
        Minimum confidence required for other classes.
    imgsz : int
        YOLO input image size.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load model
    model = YOLO(model_path)

    # Run inference
    results = model.predict(
        source=input_folder,
        save=False,
        imgsz=imgsz,
        conf=min(face_conf_thresh, other_conf_thresh)
    )

    for result in results:
        img = result.orig_img.copy()
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names

        for box, score, cls_id in zip(boxes, scores, classes):
            cls_id = int(cls_id)

            # Custom thresholding
            if cls_id == 4:  # face
                if score < face_conf_thresh:
                    continue
            else:
                if score < other_conf_thresh:
                    continue

            x1, y1, x2, y2 = map(int, box)
            label = f"{names[cls_id]} {score:.2f}"

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        # Save output
        save_path = os.path.join(output_folder, os.path.basename(result.path))
        cv2.imwrite(save_path, img)

    print(f"✓ Batch inference completed! Results saved to: {output_folder}/")


# Entry point
if __name__ == "__main__":
    run_batch_inference(
        model_path="runs/detect_yolov8s_noflip/weights/best.pt",
        input_folder="test_images",
        output_folder="batch_inference_output",
        face_conf_thresh=0.45,
        other_conf_thresh=0.25,
        imgsz=640
    )
