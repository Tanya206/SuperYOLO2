import os
import numpy as np
from glob import glob

def rotated_to_horizontal_bbox(x, y, w, h, theta):
    """
    Convert rotated bounding box (x, y, w, h, θ) to horizontal bbox (x_min, y_min, x_max, y_max).
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    w_half, h_half = w / 2, h / 2

    x_min = x - w_half * cos_t - h_half * sin_t
    y_min = y - w_half * sin_t + h_half * cos_t
    x_max = x + w_half * cos_t + h_half * sin_t
    y_max = y + w_half * sin_t - h_half * cos_t

    return int(x_min), int(y_min), int(x_max), int(y_max)

def get_class_mapping(dataset_name):
    """
    Returns the class mapping for NWPU_VHR10 and UCAS_AOD.
    """
    if dataset_name == "NWPU_VHR10":
        return {
            'airplane': 0, 'ship': 1, 'storage tank': 2, 'baseball diamond': 3,
            'tennis court': 4, 'basketball court': 5, 'ground track field': 6,
            'harbor': 7, 'bridge': 8, 'vehicle': 9
        }
    elif dataset_name == "UCAS_AOD":
        return {'car': 0, 'airplane': 1}
    else:
        raise ValueError("Unknown dataset!")

def transform_labels(dataset_name, label_folder, output_folder):
    """
    Converts labels to YOLO format.
    """
    os.makedirs(output_folder, exist_ok=True)
    class_map = get_class_mapping(dataset_name)

    label_files = glob(os.path.join(label_folder, "*.txt"))

    for label_file in label_files:
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_labels = []
        for line in lines:
            values = line.strip().split()
            
            if dataset_name == "NWPU_VHR10":
                # Convert oriented bbox to horizontal
                x, y, w, h, theta, class_name = float(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4]), values[5]
                x_min, y_min, x_max, y_max = rotated_to_horizontal_bbox(x, y, w, h, theta)
            else:
                # Already in horizontal format (UCAS-AOD)
                x_min, y_min, x_max, y_max, class_name = map(float, values)

            # Normalize for YOLO format
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            img_w, img_h = 1024, 1024  # Ensure this matches your dataset
            x_center /= img_w
            y_center /= img_h
            width /= img_w
            height /= img_h

            class_id = class_map[class_name]
            new_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save new labels
        output_file = os.path.join(output_folder, os.path.basename(label_file))
        with open(output_file, "w") as f:
            f.write("\n".join(new_labels) + "\n")

# Transform labels for NWPU_VHR10 and UCAS_AOD
transform_labels("NWPU_VHR10", "dataset/NWPU_VHR10/labels", "dataset/NWPU_VHR10_1024/labels")
transform_labels("UCAS_AOD", "dataset/UCAS_AOD/labels", "dataset/UCAS_AOD_1024/labels")
