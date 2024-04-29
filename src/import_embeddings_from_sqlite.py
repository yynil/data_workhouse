import argparse
import os
import shutil

def import_db_to_qdrant(db_file,output,host,port,collection_name='mycorpus_vdb'):
    from sqlite_utilities import SqliteDictWrapper
    db = SqliteDictWrapper(db_file)
    from tqdm import tqdm
    progress_bar = tqdm(len(db), desc=f'Importing keys for {db_file} to qdrant', unit='keys', unit_scale=True, unit_divisor=1024,dynamic_ncols=True)
    batch_import = 4096
    from qdrant_client import QdrantClient
    from qdrant_client import models
    uuids = []
    embeddings = []
    documents = []
    qdrant_client = qdrant_client = QdrantClient(host,prefer_grpc=True,grpc_port=port)
    output_uuid_file = os.path.join(output,os.path.basename(db_file).split('.')[0]+'_uuids.txt')
    with open(output_uuid_file,'w',encoding='UTF-8') as f:
        for uuid in db:
            f.write(f'{uuid}\n')
            # {'embedding':embedding,'document':document}
            value = db[uuid]
            embedding = value['embedding']
            document = value['document']
            uuids.append(uuid)
            embeddings.append(embedding)
            documents.append(document)
            if len(uuids) % batch_import == 0:
                points = models.Batch(ids=uuids, vectors=embeddings,payloads=[{'doc':documents[i]} for i in range(len(documents))])
                qdrant_client.upsert(collection_name=collection_name,points=points)
                uuids = []
                embeddings = []
                documents = []
            progress_bar.update(1)
    if len(uuids) > 0:
        points = models.Batch(ids=uuids, vectors=embeddings,payloads=[{'doc':documents[i]} for i in range(len(documents))])
        qdrant_client.upsert(collection_name=collection_name,points=points)
    db.close()

parser = argparse.ArgumentParser(description='build vdb from sqlite db')
parser.add_argument('--input', type=str, help='input file or directory')
parser.add_argument('--output', type=str, help='output directory')
parser.add_argument('--host', type=str,default='localhost', help='qdrant host')
parser.add_argument('--port', type=int,default=6334, help='qdrant port')
parser.add_argument('--collection_name', type=str,default='mycorpus_vdb', help='qdrant collection name')
parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use')
args = parser.parse_args()
os.makedirs(args.output,exist_ok=True)

if os.path.isfile(args.input):
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    import_db_to_qdrant(args.input,args.output,args.host,args.port,args.collection_name)
else:
    from multiprocessing import Pool
    os.makedirs(args.output,exist_ok=True)
    with Pool(args.num_processes) as pool:
        for file in os.listdir(args.input):
            db_path = os.path.join(args.input,file)
            if db_path.endswith('.db') and os.path.isfile(db_path):
                pool.apply_async(import_db_to_qdrant,args=(db_path,args.output,args.host,args.port,args.collection_name))
        pool.close()
        pool.join()