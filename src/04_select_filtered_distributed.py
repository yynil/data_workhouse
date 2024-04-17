def strategy_by_len(str_a,str_b):
    return 0 if len(str_a)>len(str_b) else 1

strategies = {
    'len':strategy_by_len
}
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='create filtered id using only one id')
    parser.add_argument('--input_file', type=str, help='input file')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--strategy', type=str, default='len', choices=['len','ngram'], help='strategy to use')

    args = parser.parse_args()
    import os
    import tqdm
    import polars as pl

    os.makedirs(args.output_dir,exist_ok=True)
    strategy = strategies[args.strategy]
    df = pl.read_csv(args.input_file)
    print(df)
    score_groups = [0,0.95,0.98,1,1.5]
    kept_ids_scores = [set(),set(),set(),set()]
    filtered_scores = [{},{},{},{}]
    progress = tqdm.tqdm(total=len(df),desc=f'filtering {args.input_file}')
    for row in df.iter_rows():
        doc = row[2]
        similar_doc = row[3]
        score = row[4]
        docs = [doc,similar_doc]
        ids = [row[0],row[1]]

        seleted_id = strategy(doc,similar_doc)
        kept_id = ids[seleted_id]
        filtered_id = ids[1-seleted_id]
        kept_doc = docs[seleted_id]
        filtered_doc = docs[1-seleted_id]
        
        for i in range(len(score_groups)):
            if score < score_groups[i]:
                group_index = i-1
                break
        filtered = filtered_scores[group_index]
        kept_ids = kept_ids_scores[group_index]
        if kept_id not in filtered:
            #We may keep this id
            if filtered_id in kept_ids:
                #this similar_id is already kept by previous rules
                #now we should filter this id instead
                filtered[filtered_id] = {'filtered_by':kept_id,'filtered_score':score,'original_doc':kept_doc,'similar_doc':filtered_doc}
                kept_ids.remove(filtered_id)
                progress.update(1)
                continue
            kept_ids.add(id)
            #It means this id is not filtered by other ids
            if filtered_id not in filtered:
                #It means similar_id is first filtered by id
                filtered[filtered_id] = {'filtered_by':kept_id,'filtered_score':score,'original_doc':kept_doc,'similar_doc':filtered_doc}
            else:
                #It means similar_id is filtered by other ids
                #we only keep one
                continue
        progress.update(1)
    progress.close()
    import csv
    input_basename = os.path.basename(args.input_file).split('.')[0]
    for i in range(len(filtered_scores)):
        filtered = filtered_scores[i]
        output_file = os.path.join(args.output_dir,f'filtered_{input_basename}_{score_groups[i]}_{score_groups[i+1]}.csv')
        uuid_file = os.path.join(args.output_dir,f'filtered_{input_basename}_{score_groups[i]}_{score_groups[i+1]}_uuids.txt')
        with open(output_file,'w',encoding='UTF-8') as f:
            writer = csv.writer(f,quotechar='"',quoting=csv.QUOTE_MINIMAL,escapechar='\\')
            uuid_file_fp = open(uuid_file,'w',encoding='UTF-8')
            writer.writerow(['id','similar_id','original_doc','similar_doc','score'])
            for key in filtered:
                writer.writerow([key,filtered[key]['filtered_by'],filtered[key]['original_doc'],filtered[key]['similar_doc'],filtered[key]['filtered_score']])
                uuid_file_fp.write(key+'\n')
            uuid_file_fp.close()