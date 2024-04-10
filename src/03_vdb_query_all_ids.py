def query_vdb(host):
    from qdrant_client import QdrantClient
    import qdrant_client.models as models
    client = QdrantClient(host)
    print('Client created')
    collection_name = 'mycorpus_vdb'
    offset = None
    cnt = 0
    print_cnt = 0
    all_ids = []
    while True:
        records,offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must_not=[
                    models.IsEmptyCondition(is_empty=models.PayloadField(key="doc"),)
                ],
            ),
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )
        queried_ids = [record.id for record in records]
        all_ids.extend(queried_ids)
        if offset is None:
            break
        cnt += len(records)
        print_cnt += len(records)
        if print_cnt >= 10000:
            print(f'found {cnt} records')
            print_cnt = 0
    print(f'found {cnt} records')
    with open('all_ids.txt','w',encoding='UTF-8') as f:
        for id in all_ids:
            f.write(id+'\n')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='query vdb to get all ids')
    parser.add_argument('--host', type=str, default='localhost',help='host of vdb')
    args = parser.parse_args()
    query_vdb(args.host)