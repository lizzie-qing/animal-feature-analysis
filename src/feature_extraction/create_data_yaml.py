"""
create_data_yaml.py
------------------------------------------------
Utility for automatically generating YOLO 'data.yaml'
for object detection datasets.

Author: Lizzie Qing
"""

import os
import yaml


def create_data_yaml(
    dataset_path: str,
    class_names: list,
    train_dir: str = "images/train",
    val_dir: str = "images/val",
    save_name: str = "data.yaml"
):
    """
    Generate a YOLO-compatible data.yaml configuration file.

    Parameters
    ----------
    dataset_path : str
        Root directory of the dataset.
    class_names : list
        List of class names, e.g. ['left_eye', 'right_eye', ...].
    train_dir : str
        Relative path to the training images folder.
    val_dir : str
        Relative path to the validation images folder.
    save_name : str
        Filename for the YAML file.
    """
    if not isinstance(class_names, list):
        raise ValueError("class_names must be a Python list.")

    # Ensure dataset directory exists
    os.makedirs(dataset_path, exist_ok=True)

    yaml_dict = {
        "path": dataset_path,
        "train": train_dir,
        "val": val_dir,
        "nc": len(class_names),
        "names": class_names
    }

    yaml_path = os.path.join(dataset_path, save_name)

    # Write YAML file
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, allow_unicode=True, sort_keys=False)

    print(f"✓ YOLO data.yaml successfully created at: {yaml_path}")


# Example execution
if __name__ == "__main__":
    dataset_path = "my_dataset"
    class_names = ["left_eye", "right_eye", "left_ear", "right_ear", "face"]

    create_data_yaml(dataset_path, class_names)
