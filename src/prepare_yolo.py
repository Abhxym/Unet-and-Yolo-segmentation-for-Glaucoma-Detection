import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_to_yolo_labels(root_dir, mode='train'):
    # We want yolo_data to be in the main drishti folder, not inside Training/Test folders
    # root_dir passed here is the specific Training/Test folder
    # We need to go up one level to get the main drishti folder
    main_root = os.path.dirname(root_dir)
    
    if mode == 'train':
        images_dir = os.path.join(root_dir, 'Training', 'Images')
        gt_dir = os.path.join(root_dir, 'Training', 'GT')
        output_dir = os.path.join(main_root, 'yolo_data', 'train')
    else:
        images_dir = os.path.join(root_dir, 'Test', 'Images')
        gt_dir = os.path.join(root_dir, 'Test', 'Test_GT')
        output_dir = os.path.join(main_root, 'yolo_data', 'val')
        
    print(f"Creating output dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    image_paths = glob.glob(os.path.join(images_dir, '**', '*.png'), recursive=True)
    
    for img_path in tqdm(image_paths, desc=f"Processing {mode}"):
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Load Image to get size
        img = Image.open(img_path)
        w, h = img.size
        
        # Load OD Mask
        if mode == 'train':
            gt_folder = os.path.join(gt_dir, base_name, 'SoftMap')
            od_mask_path = os.path.join(gt_folder, f"{base_name}_ODsegSoftmap.png")
        else:
            # Test structure might be slightly different, let's check
            # Based on previous ls, Test/Test_GT/drishtiGS_XXX/SoftMap/drishtiGS_XXX_ODsegSoftmap.png
            gt_folder = os.path.join(gt_dir, base_name, 'SoftMap')
            od_mask_path = os.path.join(gt_folder, f"{base_name}_ODsegSoftmap.png")
            
        if not os.path.exists(od_mask_path):
            print(f"Skipping {base_name}, mask not found.")
            continue
            
        mask = Image.open(od_mask_path).convert("L")
        mask = np.array(mask)
        mask = (mask > 128).astype(np.uint8) # Threshold
        
        # Find BBox
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            print(f"Skipping {base_name}, empty mask.")
            continue
            
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # YOLO Format: class x_center y_center width height (normalized)
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        x_center = xmin + bbox_w / 2
        y_center = ymin + bbox_h / 2
        
        x_center /= w
        y_center /= h
        bbox_w /= w
        bbox_h /= h
        
        # Save Label
        label_path = os.path.join(output_dir, 'labels', f"{base_name}.txt")
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center} {y_center} {bbox_w} {bbox_h}\n")
            
        # Copy Image (or symlink to save space, but copy is safer for now)
        img.save(os.path.join(output_dir, 'images', f"{base_name}.png"))

def prepare_yolo_data():
    root_dir = r"c:/Users/Admin/OneDrive/Desktop/Model/drishti"
    
    # Train Data
    convert_to_yolo_labels(os.path.join(root_dir, "Training-20211018T055246Z-001"), mode='train')
    
    # Val Data (Test set)
    convert_to_yolo_labels(os.path.join(root_dir, "Test-20211018T060000Z-001"), mode='test')
    
    # Create data.yaml
    yaml_path = os.path.join(root_dir, 'yolo_data', 'data.yaml')
    print(f"Writing yaml to {yaml_path}")
    
    yaml_content = f"""
path: {os.path.join(root_dir, 'yolo_data').replace(os.sep, '/')}
train: train/images
val: val/images

nc: 1
names: ['OpticDisc']
"""
    try:
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print("Yaml written successfully.")
    except Exception as e:
        print(f"Error writing yaml: {e}")
        
    print("YOLO data preparation complete.")

if __name__ == "__main__":
    prepare_yolo_data()
