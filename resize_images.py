import os
from glob import glob
from PIL import Image
import shutil

def resize_images(input_folder, output_folder, size):
    os.makedirs(output_folder, exist_ok=True)
    # Process .jpg and .png files
    for img_path in glob(os.path.join(input_folder, "*.jpg")) + glob(os.path.join(input_folder, "*.png")):
        img = Image.open(img_path)
        # Resize to a square of (size x size) using bilinear interpolation
        img_resized = img.resize((size, size), Image.BILINEAR)
        img_resized.save(os.path.join(output_folder, os.path.basename(img_path)))

# NWPU_VHR10: create 1024 and 512 versions for both train and test

# 1024×1024 version (for training)
nwpu_train_input = "dataset/NWPU_VHR10/images/train/"
nwpu_train_output_1024 = "dataset/NWPU_VHR10_1024/images/train/"
os.makedirs(nwpu_train_output_1024, exist_ok=True)
resize_images(nwpu_train_input, nwpu_train_output_1024, 1024)

nwpu_test_input = "dataset/NWPU_VHR10/images/test/"
nwpu_test_output_1024 = "dataset/NWPU_VHR10_1024/images/test/"
os.makedirs(nwpu_test_output_1024, exist_ok=True)
resize_images(nwpu_test_input, nwpu_test_output_1024, 1024)

# 512×512 version (for validation/testing)
nwpu_train_output_512 = "dataset/NWPU_VHR10_512/images/train/"
os.makedirs(nwpu_train_output_512, exist_ok=True)
resize_images(nwpu_train_input, nwpu_train_output_512, 512)

nwpu_test_output_512 = "dataset/NWPU_VHR10_512/images/test/"
os.makedirs(nwpu_test_output_512, exist_ok=True)
resize_images(nwpu_test_input, nwpu_test_output_512, 512)

# Copy labels (since they are normalized, they work for both resolutions)
def copy_labels(src, dst):
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        
copy_labels("dataset/NWPU_VHR10/labels", "dataset/NWPU_VHR10_1024/labels")
copy_labels("dataset/NWPU_VHR10/labels", "dataset/NWPU_VHR10_512/labels")

print("✅ NWPU_VHR10 images resized to 1024 and 512!")
