import os
import struct
import numpy as np
import h5py
import cv2
from tqdm import tqdm

#need to run (pip install opencv-python h5py tqdm)
#open CV comes with numpy already in it



def write_custom_binary(output_path, feature_matrices, label_matrices, target_shape=(120, 160)):
    """
    Writes flat arrays directly into a format readily read by a C Tensor database.
    target_shape: (height, width) to enforce strict dimension uniformity.
    """
    num_samples = len(feature_matrices)
    height, width = target_shape
    
    header_format = '>5I' 
    magic = 2060 
    channels = 1
    
    print(f"Exporting {num_samples} samples to {output_path}...")
    
    with open(output_path, 'wb') as f:
        f.write(struct.pack(header_format, magic, num_samples, channels, height, width))
        
        for img in tqdm(feature_matrices, desc="Writing Features (X)"):
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            # Ensure 32-bit float array matching Tensor data allocations
            f.write(resized_img.astype(np.float32).tobytes())
            
        for depth in tqdm(label_matrices, desc="Writing Labels (Y)"):
            resized_depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
            f.write(resized_depth.astype(np.float32).tobytes())

def process_nyu_to_bin(mat_file_path, output_bin_path, target_shape=(120, 160)):
    # Converts NYU H5 file matrices to flat binary stream 
    print("Processing NYU Depth V2...")
    with h5py.File(mat_file_path, 'r') as f:
        images = f['images'] 
        depths = f['depths']
        num_samples = images.shape[0]
        
        features = []
        labels = []
        
        for i in tqdm(range(num_samples), desc="Parsing NYU Frames"):
            img = images[i].transpose(2, 1, 0)
            depth_m = depths[i].transpose(1, 0)
            
            # 0.299R + 0.589G + 0.114B
            gray = (0.299 * img[:,:,0] + 0.589 * img[:,:,1] + 0.114 * img[:,:,2]) / 255.0
            inv_depth = np.where(depth_m > 0.0, 1.0 / depth_m, 0.0)
            
            features.append(gray)
            labels.append(inv_depth)
            
        write_custom_binary(output_bin_path, features, labels, target_shape)



def process_diode_to_bin(diode_val_dir, output_bin_path, target_shape=(120, 160)):
    #Traversing DIODE tree and compiling matched items
    print("Processing DIODE Dataset...")
    
    valid_pairs = []
    for root, _, files in os.walk(diode_val_dir):
        for file in files:
            if file.endswith('.png') and not file.endswith('_mask.png'):
                rgb_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                depth_path = os.path.join(root, f"{base_name}_depth.npy")
                mask_path = os.path.join(root, f"{base_name}_depth_mask.npy")
                if os.path.exists(depth_path) and os.path.exists(mask_path):
                    valid_pairs.append((rgb_path, depth_path, mask_path))
                    
    features = []
    labels = []
    
    for rgb_path, depth_path, mask_path in tqdm(valid_pairs, desc="Parsing DIODE Frames"):
        img = cv2.imread(rgb_path)
        count = 0
        if img is None:
            f.write(np.zeros((height, width), dtype=np.float32).tobytes())
            count += 1
            print("img is none writing zero to file for idx: {count}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        depth_m = np.load(depth_path).squeeze()
        mask = np.load(mask_path).squeeze()
        
        gray = (0.299 * img[:,:,0] + 0.589 * img[:,:,1] + 0.114 * img[:,:,2]) / 255.0
        inv_depth = np.where((mask > 0) & (depth_m > 0.0), 1.0 / depth_m, 0.0)
        
        features.append(gray)
        labels.append(inv_depth)

    write_custom_binary(output_bin_path, features, labels, target_shape)





def process_virtual_kitti_to_bin(rgb_dir, depth_dir, output_bin_path, target_shape=(120, 160)):
    # Stream-to-disk version of Virtual KITTI 2 processing to prevent OOM Killer crash
    print("Processing Virtual KITTI 2 (Stream-to-Disk)...")

    abs_rgb_dir = os.path.abspath(rgb_dir)
    abs_depth_dir = os.path.abspath(depth_dir)
    height, width = target_shape

    file_pairs = []
    for root, _, files in os.walk(abs_depth_dir):
        for file in files:
            if file.endswith('.png') and 'depth_' in file:
                depth_path = os.path.join(root, file)
                rel_path = os.path.relpath(depth_path, abs_depth_dir)
                
                rgb_rel_path = rel_path.replace('/depth/', '/rgb/')
                dir_part, filename = os.path.split(rgb_rel_path)
                rgb_filename = filename.replace('depth_', 'rgb_').replace('.png', '.jpg')
                rgb_path = os.path.join(abs_rgb_dir, dir_part, rgb_filename)
                
                if os.path.exists(rgb_path):
                    file_pairs.append((rgb_path, depth_path))

    num_samples = len(file_pairs)
    if num_samples == 0:
        print("Found 0 pairs. Double-check path match logic.")
        return

    print(f"Found {num_samples} valid frames. Exporting directly to {output_bin_path}...")

    header_format = '>5I'#big endian for my C parser
    magic = 2060
    channels = 1
    
    with open(output_bin_path, 'wb') as f:
        f.write(struct.pack(header_format, magic, num_samples, channels, height, width))
        
        x_count = 0
        y_count = 0

        for rgb_path, _ in tqdm(file_pairs, desc="Writing Features (X) to Disk"):
            img = cv2.imread(rgb_path)
            if img is None:
                #need to look out for all zeros being writen
                f.write(np.zeros((height, width), dtype=np.float32).tobytes())
                x_count += 1
                print("img is none writing zero to file for x: {x_count}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = (0.299 * img[:,:,0] + 0.589 * img[:,:,1] + 0.114 * img[:,:,2]) / 255.0
            resized_img = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
            f.write(resized_img.astype(np.float32).tobytes())
            
        for _, depth_path in tqdm(file_pairs, desc="Writing Labels (Y) to Disk"):
            depth_png = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth_png is None:
                #same as x
                f.write(np.zeros((height, width), dtype=np.float32).tobytes())
                y_count += 1 
                print("img is none writing zero to file for x: {y_count}")
                continue
                
            depth_m = depth_png.astype(np.float32) / 100.0
            inv_depth = np.where(depth_m > 0.0, 1.0 / depth_m, 0.0)
            resized_depth = cv2.resize(inv_depth, (width, height), interpolation=cv2.INTER_NEAREST)
            f.write(resized_depth.astype(np.float32).tobytes())

        if(x_count != y_count):
            print("mismatch in zeros: x: {x_count} || y: {y_count}")

    print(f"Successfully exported Virtual KITTI 2 dataset!")








if __name__ == "__main__":
    # 1. NYU Depth Run
    process_nyu_to_bin("../raw_data/nyu_depth/nyu_depth_v2_labeled.mat", "../filtered_data/nyu_training_set.bin")
    
    # 2. Virtual KITTI 2 Run
    process_virtual_kitti_to_bin("../raw_data/vkitti/rgb", "../raw_data/vkitti/depth", "../filtered_data/vkitti_synthetic_set.bin")
    
    # 3. DIODE Run
    process_diode_to_bin("../raw_data/diode/val", "../filtered_data/diode_val_set.bin")

#can add more datasets later













