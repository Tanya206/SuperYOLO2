import json
import os

def convert_coco_to_yolo(json_file, images_dir, labels_dir):
    # Load the COCO-format JSON annotations
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Build a mapping from image id to image info
    image_info = {img['id']: img for img in data['images']}
    
    # Accumulate annotations per image
    image_annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        bbox = ann['bbox']  # COCO format: [x, y, width, height]
        img_info = image_info[image_id]
        img_w = img_info['width']
        img_h = img_info['height']
        
        # Convert bbox to YOLO format: normalized x_center, y_center, width, height
        x_center = (bbox[0] + bbox[2] / 2) / img_w
        y_center = (bbox[1] + bbox[3] / 2) / img_h
        w_norm = bbox[2] / img_w
        h_norm = bbox[3] / img_h
        
        category_id = ann['category_id']  # Use as-is (adjust remapping if necessary)
        
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append((category_id, x_center, y_center, w_norm, h_norm))
    
    # Ensure the labels directory exists
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    
    # Write a YOLO-format .txt file for each image that has annotations
    for img_id, anns in image_annotations.items():
        img_info = image_info[img_id]
        base_name = os.path.splitext(img_info['file_name'])[0]
        label_file = os.path.join(labels_dir, base_name + '.txt')
        with open(label_file, 'w') as f:
            for ann in anns:
                # Format: <class_id> <x_center> <y_center> <width> <height>
                f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(*ann))
                
    print("Conversion complete! YOLO label files are saved in:", labels_dir)


if __name__ == '__main__':
    # For the training set:
    train_json = "dataset/NWPU_VHR10/annotations/train.json"
    train_images_dir = "dataset/NWPU_VHR10/images/train"
    train_labels_dir = "dataset/NWPU_VHR10_1024/labels/train"
    os.makedirs(train_labels_dir, exist_ok=True)
    convert_coco_to_yolo(train_json, train_images_dir, train_labels_dir)
    
    # For the test set:
    test_json = "dataset/NWPU_VHR10/annotations/test.json"
    test_images_dir = "dataset/NWPU_VHR10/images/test"
    test_labels_dir = "dataset/NWPU_VHR10_1024/labels/test"
    os.makedirs(test_labels_dir, exist_ok=True)
    convert_coco_to_yolo(test_json, test_images_dir, test_labels_dir)
