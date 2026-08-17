import argparse
import os
from tqdm import tqdm

def process_file(input_path, output_path, mode='melody'):
    """处理单个文件"""
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        skip_lines = False
        for line in infile:
            line = line.strip()
            if mode == 'melody':
                if line.startswith('Track_Chord'):
                    skip_lines = True
                    continue
                elif line.startswith('Track_Melody'):
                    skip_lines = False
            elif mode == 'perf':
                if line.startswith('Track_Melody'):
                    skip_lines = True
                    continue
                elif line.startswith('Track_Performance'):
                    skip_lines = False
            
            if not skip_lines and line:  # 只写入非跳过行且非空行
                outfile.write(line + '\n')

def process_directory(input_dir, output_dir, mode='melody'):
    """处理目录下的所有txt文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入目录中的所有文件
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_file(input_path, output_path, mode=mode)
            # print(f"处理完成: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Extract conditional tracks from generated event txt files.')
    parser.add_argument('-i', '--input_dir', default='generation/stage2_melody',
                        help='directory containing generated txt files')
    parser.add_argument('-o', '--output_dir', default='generation/stage2_melody_conn',
                        help='directory to write filtered txt files')
    parser.add_argument('-m', '--mode', choices=['melody', 'perf'], default='melody',
                        help='track filtering mode')
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, mode=args.mode)

if __name__ == '__main__':
    main()
