def strategy_by_len(str_a,str_b):
    return 0 if len(str_a)>len(str_b) else 1

strategies = {
    'len':strategy_by_len
}

from redis_utility import get_redis_client,get_docs_from_redis,get_doc_from_redis
import os
import tqdm
import polars as pl
def filter_file(input_file,output_dir,strategy,redis_host,redis_port,redis_db):
    input_dir = os.path.dirname(input_file)
    assert input_dir != output_dir
    os.makedirs(output_dir,exist_ok=True)
    strategy = strategies[strategy]
    df = pl.read_csv(input_file)
    print(df)
    score_groups = [0,0.95,0.98,1,1.5]
    kept_ids_scores = [set(),set(),set(),set()]
    filtered_scores = [{},{},{},{}]
    progress = tqdm.tqdm(total=len(df),desc=f'filtering {input_file}')
    redis_client = get_redis_client(redis_host,redis_port,redis_db)
    for row in df.iter_rows():
        progress.update(1)
        # doc = row[2]
        # similar_doc = row[3]
        score = row[4]
        # docs = [doc,similar_doc]
        ids = [row[0],row[1]]
        docs = get_docs_from_redis(redis_client,ids)
        seleted_id = strategy(docs[0],docs[1])
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
    progress.close()
    import csv
    input_basename = os.path.basename(input_file).split('.')[0]
    print(f'input_basename:{input_basename}, saved to {output_dir}')
    for i in range(len(filtered_scores)):
        filtered = filtered_scores[i]
        output_file = os.path.join(output_dir,f'filtered_{input_basename}_{score_groups[i]}_{score_groups[i+1]}.csv')
        uuid_file = os.path.join(output_dir,f'filtered_{input_basename}_{score_groups[i]}_{score_groups[i+1]}_uuids.txt')
        with open(output_file,'w',encoding='UTF-8') as f:
            writer = csv.writer(f,quotechar='"',quoting=csv.QUOTE_MINIMAL,escapechar='\\')
            uuid_file_fp = open(uuid_file,'w',encoding='UTF-8')
            writer.writerow(['id','similar_id','original_doc','similar_doc','score'])
            for key in filtered:
                writer.writerow([key,filtered[key]['filtered_by'],filtered[key]['original_doc'],filtered[key]['similar_doc'],filtered[key]['filtered_score']])
                uuid_file_fp.write(key+'\n')
            uuid_file_fp.close()
        print(f'output_file:{output_file}')
        print(f'uuid_file:{uuid_file}')
    print('done')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='create filtered id using only one id')
    #--input_file /home/yueyulin/data/wudao_uuids/filtered_0/part-2021012516_uuids_duplicated.csv --output_dir /home/yueyulin/tmp --strategy len --redis_host 192.168.1.36
    #--input_file /home/yueyulin/data/wudao_uuids/ --output_dir /home/yueyulin/tmp/tmp --strategy len --redis_host 192.168.1.36
    parser.add_argument('--input_file', type=str, help='input file')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--strategy', type=str, default='len', choices=['len','ngram'], help='strategy to use')
    parser.add_argument('--redis_host', type=str, default='localhost', help='redis host')
    parser.add_argument('--redis_port', type=int, default=6379, help='redis port')
    parser.add_argument('--redis_db', type=int, default=0, help='redis db')
    parser.add_argument('--num_process', type=int, default=4, help='number of process to use for multiprocessing')

    args = parser.parse_args()
    if os.path.isfile(args.input_file):
        filter_file(args.input_file,args.output_dir,args.strategy,args.redis_host,args.redis_port,args.redis_db)
    elif os.path.isdir(args.input_file):
        csv_files = []
        for root,dirs,files in os.walk(args.input_file):
            for file in files:
                if not file.endswith('.csv'):
                    continue
                input_file = os.path.join(root,file)
                csv_files.append(input_file)
        print(f'processing files in {args.input_file} : {csv_files}')
        from multiprocessing import Pool
        with Pool(args.num_process) as pool:
            for input_file in csv_files:
                pool.apply_async(filter_file,args=(input_file,args.output_dir,args.strategy,args.redis_host,args.redis_port,args.redis_db))
            pool.close()
            pool.join()
    else:
        print(f'input_file:{args.input_file} not exist')
    # import os
    # import tqdm
    # import polars as pl

    # os.makedirs(args.output_dir,exist_ok=True)
    # strategy = strategies[args.strategy]
    # df = pl.read_csv(args.input_file)
    # print(df)
    # score_groups = [0,0.95,0.98,1,1.5]
    # kept_ids_scores = [set(),set(),set(),set()]
    # filtered_scores = [{},{},{},{}]
    # progress = tqdm.tqdm(total=len(df),desc=f'filtering {args.input_file}')
    # redis_client = get_redis_client(args.redis_host,args.redis_port,args.redis_db)
    # for row in df.iter_rows():
    #     progress.update(1)
    #     # doc = row[2]
    #     # similar_doc = row[3]
    #     score = row[4]
    #     # docs = [doc,similar_doc]
    #     ids = [row[0],row[1]]
    #     docs = get_docs_from_redis(redis_client,ids)
    #     seleted_id = strategy(docs[0],docs[1])
    #     kept_id = ids[seleted_id]
    #     filtered_id = ids[1-seleted_id]
    #     kept_doc = docs[seleted_id]
    #     filtered_doc = docs[1-seleted_id]
        
    #     for i in range(len(score_groups)):
    #         if score < score_groups[i]:
    #             group_index = i-1
    #             break
    #     filtered = filtered_scores[group_index]
    #     kept_ids = kept_ids_scores[group_index]
    #     if kept_id not in filtered:
    #         #We may keep this id
    #         if filtered_id in kept_ids:
    #             #this similar_id is already kept by previous rules
    #             #now we should filter this id instead
    #             filtered[filtered_id] = {'filtered_by':kept_id,'filtered_score':score,'original_doc':kept_doc,'similar_doc':filtered_doc}
    #             kept_ids.remove(filtered_id)
    #             continue
    #         kept_ids.add(id)
    #         #It means this id is not filtered by other ids
    #         if filtered_id not in filtered:
    #             #It means similar_id is first filtered by id
    #             filtered[filtered_id] = {'filtered_by':kept_id,'filtered_score':score,'original_doc':kept_doc,'similar_doc':filtered_doc}
    #         else:
    #             #It means similar_id is filtered by other ids
    #             #we only keep one
    #             continue
    # progress.close()
    # import csv
    # input_basename = os.path.basename(args.input_file).split('.')[0]
    # print(f'input_basename:{input_basename}, saved to {args.output_dir}')
    # for i in range(len(filtered_scores)):
    #     filtered = filtered_scores[i]
    #     output_file = os.path.join(args.output_dir,f'filtered_{input_basename}_{score_groups[i]}_{score_groups[i+1]}.csv')
    #     uuid_file = os.path.join(args.output_dir,f'filtered_{input_basename}_{score_groups[i]}_{score_groups[i+1]}_uuids.txt')
    #     with open(output_file,'w',encoding='UTF-8') as f:
    #         writer = csv.writer(f,quotechar='"',quoting=csv.QUOTE_MINIMAL,escapechar='\\')
    #         uuid_file_fp = open(uuid_file,'w',encoding='UTF-8')
    #         writer.writerow(['id','similar_id','original_doc','similar_doc','score'])
    #         for key in filtered:
    #             writer.writerow([key,filtered[key]['filtered_by'],filtered[key]['original_doc'],filtered[key]['similar_doc'],filtered[key]['filtered_score']])
    #             uuid_file_fp.write(key+'\n')
    #         uuid_file_fp.close()
    #     print(f'output_file:{output_file}')
    #     print(f'uuid_file:{uuid_file}')
    # print('done')