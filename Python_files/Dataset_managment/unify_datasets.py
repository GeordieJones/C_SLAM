import os
import struct
import numpy as np
from tqdm import tqdm

def merge_and_shuffle_binaries(input_paths, output_path):
    """
    Combines individual binary datasets and shuffles them globally without loading 
    raw image arrays into memory, preventing Out-Of-Memory (OOM) crashes.
    """
    header_format = '>5I'     
    header_size = struct.calcsize(header_format)
    
    total_samples = 0
    unified_channels = None
    unified_height = None
    unified_width = None
    
    # Layout of each tuple: (file_index, local_frame_index)
    virtual_index_map = []
    
    print("Pre-scanning binaries to construct a virtual index map...")
    for file_idx, path in enumerate(input_paths):
        if not os.path.exists(path):
            print(f"Warning: File {path} not found. Skipping.")
            continue
            
        print(f" -> Mapping layout of: {path}")
        with open(path, 'rb') as f:
            header_bytes = f.read(header_size)
            magic, num_samples, channels, height, width = struct.unpack(header_format, header_bytes)
            
            if magic != 2060:
                raise ValueError(f"Malformed file or wrong magic number in {path}")
                
            if unified_channels is None:
                unified_channels = channels
                unified_height = height
                unified_width = width
            else:
                if (channels != unified_channels or height != unified_height or width != unified_width):
                    raise ValueError(f"Dimension mismatch in {path}.")
            
            for frame_idx in range(num_samples):
                virtual_index_map.append((file_idx, frame_idx, num_samples))
                
            total_samples += num_samples

    if total_samples == 0:
        print("Error: No data accumulated.")
        return

    frame_elements = unified_channels * unified_height * unified_width
    frame_bytes_size = frame_elements * 4 # float32 = 4 bytes per pixel element

    print(f"\nSuccessfully mapped {total_samples} samples across domains.")
    print("Performing global shuffle on the lookup table...")
    
    np.random.shuffle(virtual_index_map)
    
    print(f"Writing master dataset to {output_path} via disk-streaming streams...")
    
    source_files = [open(path, 'rb') for path in input_paths if os.path.exists(path)]
    
    try:
        with open(output_path, 'wb') as out_f:
            out_f.write(struct.pack(header_format, 2060, total_samples, unified_channels, unified_height, unified_width))
            
            for file_idx, frame_idx, _ in tqdm(virtual_index_map, desc="Streaming Master Features (X)"):
                src_f = source_files[file_idx]
                offset = header_size + (frame_idx * frame_bytes_size)
                src_f.seek(offset)
                out_f.write(src_f.read(frame_bytes_size))
                
            for file_idx, frame_idx, num_samples in tqdm(virtual_index_map, desc="Streaming Master Labels (Y)"):
                src_f = source_files[file_idx]
                total_features_size = num_samples * frame_bytes_size
                offset = header_size + total_features_size + (frame_idx * frame_bytes_size)
                src_f.seek(offset)
                out_f.write(src_f.read(frame_bytes_size))
                
    finally:
        for src_f in source_files:
            src_f.close()
            
    print("\nCompilation complete! Master file generated safely without using RAM.")

if __name__ == "__main__":
    datasets_to_stitch = [
        "../filtered_data/nyu_training_set.bin",
        "../filtered_data/diode_val_set.bin",
        "../filtered_data/vkitti_synthetic_set.bin"
    ]
    
    merge_and_shuffle_binaries(datasets_to_stitch, "../filtered_data/master_dataset.bin")







