import colorama
from grpc import Compression
def query_vdb_find_candidate(host,id_file,score_threshold,output_dir,collection_name,fetch_doc=False):
    print(colorama.Fore.GREEN + f"querying vdb for {id_file}, with threshold {score_threshold}, with output {output_dir} to host {host}, if fetch doc {fetch_doc}" + colorama.Style.RESET_ALL)
    with open(id_file,'r',encoding='UTF-8') as f:
        ids = f.readlines()
    from qdrant_client import QdrantClient
    import qdrant_client.models as models
    client = QdrantClient(host,prefer_grpc=True,grpc_port=6334,http2=True,grpc_compression=Compression.Gzip)
    print('Client created')
    # collection_name = 'mycorpus_vdb'
    offset = None
    cnt = 0
    print_cnt = 0
    duplicated_data = {
        'id':[],
        'similar_id':[],
        'original_doc':[],
        'similar_doc':[],
        'score':[]
    }
    from tqdm import tqdm
    progess = tqdm(ids,desc=f'querying id in {id_file}')
    for id in progess:
        id = id.strip()
        records = client.retrieve(collection_name,[id],with_payload=True,with_vectors=True)
        if len(records) == 0:
            print(f'{id} not found')
            continue
        search_params=models.SearchParams(hnsw_ef=128, exact=False)
        search_queries = [
            models.SearchRequest(
                vector=record.vector,
                filter=models.Filter(
                    must_not=[
                        models.IsEmptyCondition(is_empty=models.PayloadField(key="doc")),
                        models.HasIdCondition(has_id=[record.id]),
                    ],
                ),
                limit=10,
                params=search_params,
                score_threshold=score_threshold,
                with_payload=fetch_doc,
            )
            for record in records
        ]
        results = client.search_batch(collection_name=collection_name,requests=search_queries)
        for i in range(len(results)):
            result = results[i]
            if len(result) > 0:
                for r in result:
                    duplicated_data['id'].append(id)
                    duplicated_data['similar_id'].append(r.id)
                    duplicated_data['original_doc'].append(records[i].payload['doc'] if fetch_doc else '')
                    duplicated_data['similar_doc'].append(r.payload['doc'] if fetch_doc else '')
                    duplicated_data['score'].append(r.score)
    client.close()
    import pandas as pd
    import os
    df = pd.DataFrame(duplicated_data)
    output_file = os.path.join(output_dir,os.path.basename(id_file).split('.')[0]+'_duplicated.csv')
    df.to_csv(output_file,index=False,escapechar='\\')
    print(f'output saved to {output_file} for {id_file}')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='query vdb')
    parser.add_argument('--host', type=str, default='localhost',help='host of vdb')
    parser.add_argument('--id_file', type=str, default='all_ids.txt',help='id_file to query')
    parser.add_argument('--input_dir',type=str,help='input directory')
    parser.add_argument('--num_process',type=int,default=4,help='number of process to use for multiprocessing')
    parser.add_argument('--score_threshold',type=float,default=0.9,help='score threshold to consider as duplicated')
    parser.add_argument('--output_dir',type=str,help='output directory',required=True)
    parser.add_argument('--fetch_doc',action='store_true',help='fetch document',default=False)
    parser.add_argument('--collection_name',type=str,default='wudao',help='collection name')
    args = parser.parse_args()
    import os
    os.makedirs(args.output_dir,exist_ok=True)
    if args.input_dir:
        import multiprocessing as mp
        file_list = []
        for file in os.listdir(args.input_dir):
            if file.endswith('.txt'):
                file_list.append(os.path.join(args.input_dir,file))
        with mp.Pool(args.num_process) as pool:
            for file in file_list:
                pool.apply_async(query_vdb_find_candidate,args=(args.host,file,args.score_threshold,args.output_dir,args.collection_name,args.fetch_doc))
        pool.close()
        pool.join()
        print('finished')
    else:
        query_vdb_find_candidate(args.host,args.id_file,args.score_threshold,args.output_dir,args.collection_name,args.fetch_doc)
    