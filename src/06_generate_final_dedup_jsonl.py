import argparse
import os
import redis
from redis_utility import get_redis_client,get_docs_from_redis
import orjson

def generate_final_dedup_jsonl(input_file:str, output_dir:str,filtered_uuids :set,redis_host:str,redis_port:int,redis_db:int):
    os.makedirs(output_dir,exist_ok=True)
    batch_size = 1000
    redis_client = get_redis_client(redis_host,redis_port,redis_db)
    output_file = os.path.join(output_dir,os.path.basename(input_file).split('.')[0]+'_dedup.jsonl')
    of = open(output_file,'w',encoding='utf-8')
    from tqdm import tqdm
    with open(input_file,'r',encoding='utf-8') as f:
        batch_ids = []
        lines = f.readlines()
        progress = tqdm(lines,desc=f'generating dedup jsonl for {input_file}')
        for line in progress:
            uuid = line.strip()
            if uuid in filtered_uuids:
                continue
            batch_ids.append(uuid)
            if len(batch_ids) < batch_size:
                continue
            docs = get_docs_from_redis(redis_client,batch_ids)
            for i in range(len(docs)):
                doc = docs[i]
                of.write(orjson.dumps({'uuid':batch_ids[i], 'text':doc}).decode('utf-8')+'\n')
            batch_ids = []
    if len(batch_ids) > 0:
        docs = get_docs_from_redis(redis_client,batch_ids)
        for i in range(len(docs)):
            doc = docs[i]
            of.write(orjson.dumps({'uuid':batch_ids[i], 'text':doc}).decode('utf-8')+'\n')
    of.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate final dedup jsonl')
    parser.add_argument('--input', type=str, help='input file')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--filtered_uuids_file', type=str, help='filtered uuids')

    parser.add_argument('--redis_host', type=str,default='localhost', help='redis host')
    parser.add_argument('--redis_port', type=int,default=6379, help='redis port')
    parser.add_argument('--redis_db', type=int,default=0, help='redis db')

    parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use')
    args = parser.parse_args()
    if os.path.isfile(args.input):
        with open(args.filtered_uuids_file,'r') as f:
            filtered_uuids = set([line.strip() for line in f])
        generate_final_dedup_jsonl(args.input,args.output_dir,filtered_uuids,args.redis_host,args.redis_port,args.redis_db)
    elif os.path.isdir(args.input):
        from multiprocessing import Pool
        files = []
        for root,dirs,file in os.walk(args.input):
            for f in file:
                if f.endswith('_uuids.txt'):
                    files.append(os.path.join(root,f))
        print(f'processing files in {args.input} : {files}')
        with Pool(args.num_processes) as pool:
            with open(args.filtered_uuids_file,'r') as f:
                filtered_uuids = set([line.strip() for line in f])
            for file in files:
                pool.apply_async(generate_final_dedup_jsonl,args=(file,args.output_dir,filtered_uuids,args.redis_host,args.redis_port,args.redis_db))
            pool.close()
            pool.join()