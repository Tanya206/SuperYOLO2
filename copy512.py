import shutil
shutil.copytree("dataset/NWPU_VHR10_1024/labels", "dataset/NWPU_VHR10_512/labels", dirs_exist_ok=True)
print("Labels copied to NWPU_VHR10_512!")
