import os
import numpy as np
from PIL import Image

def get_vertical_diameter(mask_2d):
    rows = np.any(mask_2d, axis=1)
    if not np.any(rows):
        return 0
    ymin, ymax = np.where(rows)[0][[0, -1]]
    return ymax - ymin

def check_gt_vcdr():
    root_dir = r"c:/Users/Admin/OneDrive/Desktop/Model/drishti/Test-20211018T060000Z-001/Test/Test_GT/drishtiGS_001/SoftMap"
    od_path = os.path.join(root_dir, "drishtiGS_001_ODsegSoftmap.png")
    cup_path = os.path.join(root_dir, "drishtiGS_001_cupsegSoftmap.png")
    
    if not os.path.exists(od_path):
        print("GT files not found.")
        return

    od_mask = np.array(Image.open(od_path).convert("L"))
    cup_mask = np.array(Image.open(cup_path).convert("L"))
    
    od_mask = (od_mask > 128)
    cup_mask = (cup_mask > 128)
    
    od_h = get_vertical_diameter(od_mask)
    cup_h = get_vertical_diameter(cup_mask)
    
    vcdr = cup_h / od_h if od_h > 0 else 0
    print(f"Image: drishtiGS_001")
    print(f"GT OD Height: {od_h}")
    print(f"GT Cup Height: {cup_h}")
    print(f"GT vCDR: {vcdr:.4f}")

if __name__ == "__main__":
    check_gt_vcdr()
