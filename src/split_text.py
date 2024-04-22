import argparse
import os
import multiprocessing as mp

def split_large_jsonl(input_file_path, max_lines_per_file, output_folder):
    """
    Splits a large JSONL file into smaller files containing at most max_lines_per_file lines each.
    The resulting files are saved in the specified output folder, with filenames based on the input file name and an index.

    Args:
        input_file_path (str): Path to the large JSONL file.
        max_lines_per_file (int): Maximum number of lines per output file.
        output_folder (str): Path to the folder where the smaller JSONL files will be saved.
    """

    line_count = 0
    current_file_index = 1
    input_filename_without_extension = os.path.splitext(os.path.basename(input_file_path))[0]
    current_output_file_path = os.path.join(output_folder, f"{input_filename_without_extension}_{current_file_index:05d}.jsonl")

    with open(input_file_path, 'r', encoding='UTF-8') as jsonl_file:
        with open(current_output_file_path, 'w', encoding='UTF-8') as current_output_file:
            for line in jsonl_file:
                current_output_file.write(line)
                line_count += 1

                if line_count == max_lines_per_file:
                    line_count = 0
                    current_file_index += 1
                    current_output_file_path = os.path.join(output_folder, f"{input_filename_without_extension}_{current_file_index:05d}.jsonl")
                    current_output_file = open(current_output_file_path, 'w', encoding='UTF-8')

    # Close the last output file if it has fewer than max_lines_per_file lines
    if line_count < max_lines_per_file:
        current_output_file.close()

def main():
    parser = argparse.ArgumentParser(description="Split JSONL files in the input directory into smaller chunks and save them in the output directory.")
    parser.add_argument("--input", help="Path to the JSONL file or directory containing JSONL files to split.")
    parser.add_argument("--output_dir", help="Path to the directory where the split JSONL files will be saved.")
    parser.add_argument("--max-lines-per-file", type=int, default=232772, help="Maximum number of lines per output file (default: 232772).")
    parser.add_argument("--num-processes", type=int, default=8, help="Number of processes to use when input is a directory (default: 8).")

    args = parser.parse_args()

    max_lines_per_file = args.max_lines_per_file
    input_path = args.input
    output_dir = args.output_dir
    num_processes = args.num_processes

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    is_dir = os.path.isdir(input_path)

    if is_dir:
        with mp.Pool(num_processes) as pool:
            file_list = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.jsonl')]
            for file_path in file_list:
                pool.apply_async(split_large_jsonl, args=(file_path, max_lines_per_file, output_dir))
            pool.close()
            pool.join()
    else:
        if os.path.isfile(input_path) and input_path.endswith('.jsonl'):
            split_large_jsonl(input_path, max_lines_per_file, output_dir)

if __name__ == "__main__":
    main()