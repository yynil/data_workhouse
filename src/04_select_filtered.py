import os
import colorama
import pandas as pd
def build_data_in_memory(input_dir):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
    print(colorama.Fore.GREEN + f'found {files} files' + colorama.Style.RESET_ALL)
    # 初始化一个空的dataframe列表
    dataframes = []

    # 遍历files文件夹中的所有文件
    for filename in files:
        try:
            # 尝试读取文件为dataframe
            df = pd.read_csv(filename, escapechar='\\')
                
            # 检查是否所有需要的列都存在
            if set(['id', 'similar_id', 'original_doc', 'similar_doc', 'score']).issubset(df.columns):
                # 如果文件格式正确，添加到dataframe列表中
                dataframes.append(df)
        except pd.errors.ParserError:
            # 如果文件不能被解析为CSV，忽略它
            pass

    # 合并所有的dataframe
    final_df = pd.concat(dataframes, ignore_index=True)
    return final_df
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='select filtered')
    parser.add_argument('--input_dir', default='/home/yueyulin/tmp/skypile_splitted',type=str, help='input directory')
    parser.add_argument('--output_dir', default='/home/yueyulin/tmp/skypile_filtered',type=str, help='output directory')
    args = parser.parse_args()

    data_frame = build_data_in_memory(args.input_dir)
    print(data_frame)

    score_groups = [0,0.95,0.98,1,1.5]
    filtered_scores = [{},{},{},{}]
    kept_ids_scores = [set(),set(),set(),set()]
    from tqdm import tqdm
    progress = tqdm(total=len(data_frame),desc='filtering')
    for index,row in data_frame.iterrows():
        id = row['id']
        similar_id = row['similar_id']
        score = row['score']
        group_index = 0
        for i in range(len(score_groups)):
            if score < score_groups[i]:
                group_index = i-1
                break
        filtered = filtered_scores[group_index]
        kept_ids = kept_ids_scores[group_index]
        if id not in filtered:
            if similar_id in kept_ids:
                #this similar_id is already kept by previous rules
                #we should filter this id instead
                filtered[id] = {'filtered_by':[similar_id],'filtered_score':[score],'original_doc':[row['similar_doc']],'similar_doc':[row['original_doc']]}
                continue
            kept_ids.add(id)
            #It means this id is not filtered by other ids
            if similar_id not in filtered:
                #It means similar_id is first filtered by id
                filtered[similar_id] = {'filtered_by':[id],'filtered_score':[score],'original_doc':[row['original_doc']],'similar_doc':[row['similar_doc']]}
            else:
                #It means similar_id is filtered by other ids
                #we only keep one
                continue
        progress.update(1)
    progress.close()
    os.makedirs(args.output_dir,exist_ok=True)
    import csv
    
    for i in range(len(filtered_scores)):
        filtered = filtered_scores[i]
        output_file = os.path.join(args.output_dir,f'filtered_{score_groups[i]}_{score_groups[i+1]}.csv')
        print(colorama.Fore.GREEN+f'saving to {output_file}'+colorama.Style.RESET_ALL)
        writer = csv.DictWriter(open(output_file,'w',encoding='UTF-8'),fieldnames=['id','filtered_by','filtered_score','original_doc','similar_doc'])
        writer.writeheader()
        for key in filtered:
            data = filtered[key]
            result_len = len(data['filtered_by'])
            for j in range(result_len):
                writer.writerow({'id':key,'filtered_by':data['filtered_by'][j],'filtered_score':data['filtered_score'][j],'original_doc':data['original_doc'][j],'similar_doc':data['similar_doc'][j]})
        print(f'output saved to {output_file}')
        meta = f'filtered_{score_groups[i]}_{score_groups[i+1]} with {len(filtered)} records'
        meta_file = os.path.join(args.output_dir,f'meta_{score_groups[i]}_{score_groups[i+1]}.txt')
        print(colorama.Fore.GREEN+f'saving meta to {meta_file}'+colorama.Style.RESET_ALL)
        print(meta)
        with open(meta_file,'w',encoding='UTF-8') as f:
            f.write(meta)