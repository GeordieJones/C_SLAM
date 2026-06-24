import os
import struct
import numpy as np
from tqdm import tqdm

def reshuffle_master_binary(input_master_path, output_master_path):
    """
    Reads an existing master binary dataset, shuffles all sample pairs globally, 
    and streams out a newly randomized master dataset using 0 MB of image RAM.
    """
    header_format = '>5I'
    header_size = struct.calcsize(header_format)
    
    if not os.path.exists(input_master_path):
        print(f"Error: Source file {input_master_path} not found.")
        return

    print(f"Reading header metadata from {input_master_path}...")
    with open(input_master_path, 'rb') as f:
        header_bytes = f.read(header_size)
        magic, total_samples, channels, height, width = struct.unpack(header_format, header_bytes)
        
    if magic != 2060:
        raise ValueError("Malformed file or wrong magic number.")

    frame_bytes_size = channels * height * width * 4
    
    print(f"Found {total_samples} total samples. Generating virtual index mapping...")
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    print(f"Streaming reshuffled data into {output_master_path}...")
    with open(input_master_path, 'rb') as src_f, open(output_master_path, 'wb') as out_f:
        out_f.write(struct.pack(header_format, magic, total_samples, channels, height, width))
        
        for idx in tqdm(indices, desc="Reshuffling Features (X)"):
            offset = header_size + (idx * frame_bytes_size)
            src_f.seek(offset)
            out_f.write(src_f.read(frame_bytes_size))
            
        total_features_block_size = total_samples * frame_bytes_size
        for idx in tqdm(indices, desc="Reshuffling Labels (Y)"):
            offset = header_size + total_features_block_size + (idx * frame_bytes_size)
            src_f.seek(offset)
            out_f.write(src_f.read(frame_bytes_size))

    print("\nReshuffle complete! New randomized master dataset generated successfully.")

if __name__ == "__main__":
    reshuffle_master_binary(
        "../filtered_data/master_dataset.bin", 
        "../filtered_data/master_dataset_reshuffled.bin"
    )















