if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='merge filtered uuids')
    parser.add_argument('--input_dir', default='/home/yueyulin/tmp/tmp/',type=str, help='input directory')
    parser.add_argument('--output_dir', default='/home/yueyulin/tmp/',type=str, help='output file')
    parser.add_argument('--select_min',type=float,default=0.97,help='select min score')
    args = parser.parse_args()

    import os
    files = os.listdir(args.input_dir)
    import regex as re
    pattern = re.compile(r'filtered_(.+)_uuids_duplicated_([0-9\.]+)_([0-9\.]+)_uuids.txt')
    uuids = set()
    for file in files:
        m = pattern.match(file)
        if m:
            file_name = m.group(1)
            score_threshold_down = m.group(2)
            score_threshold_up = m.group(3)
            print(f'file_name:{file_name},score_threshold_down:{score_threshold_down},score_threshold_up:{score_threshold_up}')
            if float(score_threshold_down) < args.select_min:
                print(f'skip {file} because score_threshold_down < {args.select_min}')
                continue            
            print(f'processing {file}')
            with open(os.path.join(args.input_dir,file),'r') as f:
                for line in f:
                    uuids.add(line.strip())

    output_file = os.path.join(args.output_dir,f'filtered_{args.select_min}.txt')
    with open(output_file,'w') as f:
        for uuid in uuids:
            f.write(f'{uuid}\n')
    print(f'output saved to {output_file}, the whole uuids count is {len(uuids)} for minimum score {args.select_min}')