def query_vdb_find_candidate(host,id_file,score_threshold=0.9):
    with open(id_file,'r',encoding='UTF-8') as f:
        ids = f.readlines()
    from qdrant_client import QdrantClient
    import qdrant_client.models as models
    client = QdrantClient(host)
    print('Client created')
    collection_name = 'mycorpus_vdb'
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
    progess = tqdm(ids,desc='querying id')
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
                with_payload=True,
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
                    duplicated_data['original_doc'].append(records[i].payload['doc'])
                    duplicated_data['similar_doc'].append(r.payload['doc'])
                    duplicated_data['score'].append(r.score)
    import pandas as pd
    import os
    df = pd.DataFrame(duplicated_data)
    output_file = os.path.join(os.path.dirname(id_file),os.path.basename(id_file).split('.')[0]+'_duplicated.csv')
    df.to_csv(output_file,index=False)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='query vdb')
    parser.add_argument('--host', type=str, default='localhost',help='host of vdb')
    parser.add_argument('--id_file', type=str, default='all_ids.txt',help='id_file to query')
    args = parser.parse_args()
    query_vdb_find_candidate(args.host,args.id_file)
    