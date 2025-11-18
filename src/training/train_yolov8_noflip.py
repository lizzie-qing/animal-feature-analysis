"""
train_yolov8_noflip.py
---------------------------------------------------------
Fine-tunes a YOLOv8 model for animal facial landmark detection.
Left-right image flipping augmentation is disabled to avoid
confusing left-eye/right-eye or left-ear/right-ear labels.

Author: Lizzie Qing
"""

from ultralytics import YOLO


def train_yolov8(
    model_path: str,
    data_yaml: str,
    project_dir: str = "runs",
    run_name: str = "detect_yolov8s_noflip",
    epochs: int = 80,
    batch_size: int = 8,
    imgsz: int = 640,
    patience: int = 15
):
    """
    Fine-tune YOLOv8 with left-right flip disabled.

    Parameters
    ----------
    model_path : str
        Path to pretrained checkpoint (e.g., 'best.pt').
    data_yaml : str
        Path to data.yaml configuration.
    project_dir : str
        Root output directory for YOLO runs.
    run_name : str
        Name of this training run.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    imgsz : int
        Input image size.
    patience : int
        Early stopping patience.
    """
    model = YOLO(model_path)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,
        fliplr=0.0,        # disable left-right flip augmentation
        project=project_dir,
        name=run_name,
        verbose=True,
        save=True
    )

    print(f"✓ Training complete! Run saved under: {project_dir}/{run_name}/")


if __name__ == "__main__":
    train_yolov8(
        model_path="runs/detect_yolov8s_finetune_v2/weights/best.pt",
        data_yaml="my_dataset/data.yaml",
        project_dir="runs",
        run_name="detect_yolov8s_noflip",
        epochs=80,
        batch_size=8,
        imgsz=640,
        patience=15
    )
