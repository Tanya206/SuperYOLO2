import os
import glob

def create_path_txt(image_dir, txt_file):
    # Assume image_dir is something like "dataset/NWPU_VHR10_1024/images/train"
    base = os.path.basename(image_dir)  # "train"
    img_files = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))
    img_files = sorted(img_files)
    with open(txt_file, 'w') as f:
        for img in img_files:
            # Write relative path from the parent of image_dir (i.e., just "train/filename.jpg")
            rel_path = os.path.join(base, os.path.basename(img))
            rel_path = rel_path.replace(os.sep, '/')
            f.write(rel_path + '\n')
    print(f"Created {txt_file} with {len(img_files)} paths.")

# For training (1024 version)
create_path_txt("dataset/NWPU_VHR10_1024/images/train", "dataset/NWPU_VHR10_1024/train.txt")
# For validation (512 version)
create_path_txt("dataset/NWPU_VHR10_512/images/test", "dataset/NWPU_VHR10_512/test.txt")
