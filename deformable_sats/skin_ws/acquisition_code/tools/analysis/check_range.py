import struct
import json
import sys
import os

def main():
    # 검사할 .bin 경로 (저장소 상대 예시; 실제 파일로 교체)
    file_path = 'skin_ws/raw_data/<test_folder>/ethermotion_encoder_*.bin'

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    with open(file_path, 'rb') as f:
        # Read header
        magic = f.readline() # Magic
        header_json = f.readline()
        end_header = f.readline() # END_HEADER

        data = f.read()

    # ETHERMOTION_RECORD_STRUCT = struct.Struct('<Qddddiiii')
    # elapsed_ns, x_cmd, y_cmd, z_cmd, u_cmd, x_lcmd, y_lcmd, z_lcmd, u_lcmd
    record_struct = struct.Struct('<Qddddiiii')
    record_size = record_struct.size

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    # Iterate over the records
    # The file can be very large, but we need to check the full range
    for i in range(0, len(data) - record_size + 1, record_size):
        try:
            record = record_struct.unpack_from(data, i)
            x_cmd = record[1]
            y_cmd = record[2]
            
            if x_cmd < min_x: min_x = x_cmd
            if x_cmd > max_x: max_x = x_cmd
            if y_cmd < min_y: min_y = y_cmd
            if y_cmd > max_y: max_y = y_cmd
        except struct.error:
            break

    print(f'X Range: {min_x:.4f} to {max_x:.4f}')
    print(f'Y Range: {min_y:.4f} to {max_y:.4f}')

if __name__ == "__main__":
    main()
