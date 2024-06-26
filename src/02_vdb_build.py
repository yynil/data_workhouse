import os
import sys
parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rwkv_ext_dir = os.path.join(parent_parent_dir, 'RWKV_LM_EXT')
sys.path.append(rwkv_ext_dir)
print(f'added {rwkv_ext_dir} to sys.path')
import chromadb.db
import colorama

import argparse
import chromadb
import orjson
import tqdm

import multiprocessing as mp
from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams
from qdrant_client import models
import traceback

def add_record_to_db(input_file, id_field, content_field,rwkv_base,lora_path,is_qdrant,use_bge=False,bge_path=None,need_clean=False,host='localhost',batch_size=8):
    import uuid
    import os
    if need_clean:
        from clean import clean_in_order
        clean_functions=[
                    'bullets',
                    'dashes',
                    'extra_whitespace',
                    'ordered_bullets',
                    'group_broken_paragraphs',
                    'replace_unicode_quotes'
                 ]
    if use_bge:
        from FlagEmbedding import BGEM3FlagModel
        encoder = BGEM3FlagModel(bge_path)
    else:
        import os
        import sys
        parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        rwkv_ext_dir = os.path.join(parent_parent_dir, 'RWKV_LM_EXT')
        sys.path.append(rwkv_ext_dir)
        print(f'added {rwkv_ext_dir} to sys.path')
        from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
        import inspect
        file_path = inspect.getfile(TRIE_TOKENIZER)
        dir_name = os.path.dirname(file_path)
        tokenizer_file = os.path.join(dir_name, 'rwkv_vocab_v20230424.txt')
        tokenizer = TRIE_TOKENIZER(tokenizer_file)
        import torch
        from src.model_run import BiEncoder, RWKV,create_empty_args,load_embedding_ckpt_and_parse_args
        dtype = torch.bfloat16
        args = create_empty_args()
        w = load_embedding_ckpt_and_parse_args(rwkv_base, args)
        model = RWKV(args)
        info = model.load_state_dict(w)
        model = model.to(dtype)
        model.eval()
        print(f'loaded model from {rwkv_base}, info: {info}')
        encoder = BiEncoder(model,lora_path,tokenizer,dtype=dtype,lora_type='lora',add_mlp=True,mlp_dim=1024,lora_r=8,lora_alpha=32,target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],adapter_name='bi_embedding_lora',original_adapter_name='embedding_lora')
        # from infer.encoders import BiEncoder
        # encoder = BiEncoder(rwkv_base,lora_path,tokenizer)
    print(colorama.Fore.GREEN + f"adding records from {input_file}" + colorama.Style.RESET_ALL)
    batch_insert = 1000
    all_uuids_added = []
    try:
        with open(input_file, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            if is_qdrant:
                qdrant_client = QdrantClient(host,prefer_grpc=True,grpc_port=6334)
            else:
                chroma_client = chromadb.HttpClient(host=host, port=8000)
                chroma_collection = chroma_client.get_or_create_collection('mycorpus_vdb')
            progress_bar = tqdm.tqdm(lines,desc=f'adding {input_file} to vdb')
            documents=[]
            all_embeddings=[]
            for line in progress_bar:
                data_obj = orjson.loads(line)
                content = data_obj[content_field]
                if need_clean:
                    content = clean_in_order(content,clean_functions)
                if content is None or len(content) == 0:
                    continue
                if use_bge:
                    embeddings = encoder.encode([content],batch_size=1,max_length=2048)['dense_vecs'].tolist()
                else:
                    embeddings = [encoder.encode_texts(content).tolist()]
                documents.append(content)
                all_embeddings.extend(embeddings)
                if len(documents) >= batch_insert:
                    print(colorama.Fore.YELLOW+f'adding {batch_insert} records to vdb'+colorama.Style.RESET_ALL)
                    if is_qdrant:
                        uuids = [str(uuid.uuid4()) for i in range(len(documents))]
                        assert len(uuids) == len(all_embeddings) and len(uuids) == len(documents)
                        for i in range(len(documents)):
                            assert len(all_embeddings[i]) == 1024
                            assert len(documents[i]) > 0
                        points = models.Batch(ids=uuids, vectors=all_embeddings,payloads=[{'doc':documents[i]} for i in range(len(documents))])
                        qdrant_client.upsert(collection_name='mycorpus_vdb',points=points)
                        all_uuids_added.extend(uuids)
                    else:
                        chroma_collection.add(documents=documents,embeddings=all_embeddings,ids=uuids)
                    documents=[]
                    all_embeddings=[]
            if len(documents) > 0:
                if is_qdrant:
                    uuids = [str(uuid.uuid4()) for i in range(len(documents))]
                    assert len(uuids) == len(all_embeddings) and len(uuids) == len(documents)
                    for i in range(len(documents)):
                        assert len(all_embeddings[i]) == 1024
                        assert len(documents[i]) > 0
                    points = models.Batch(ids=uuids, vectors=all_embeddings,payloads=[{'doc':documents[i]} for i in range(len(documents))])
                    qdrant_client.upsert(collection_name='mycorpus_vdb',points=points)
                    all_uuids_added.extend(uuids)
                else:
                    chroma_collection.add(documents=documents,embeddings=embeddings,ids=uuids)
        #uuid is stored into the basename(input_file)_uuids.txt
        uuid_file = os.path.join(os.path.dirname(input_file),os.path.basename(input_file).split('.')[0]+'_uuids.txt')
        print(colorama.Fore.YELLOW+f'saving uuids to {uuid_file}'+colorama.Style.RESET_ALL)
        with open(uuid_file,'w',encoding='UTF-8') as f:
            for uuid in all_uuids_added:
                f.write(uuid+'\n')
        print(colorama.Fore.RED+f'finished reading {input_file}'+colorama.Style.RESET_ALL)
    except Exception as e:
        traceback.print_exc()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build vdb from corpus')
    parser.add_argument('--input', type=str, help='input file or directory')
    parser.add_argument('--output', type=str, help='output directory')
    parser.add_argument('--id_field', type=str,default='ID', help='field to use as id')
    parser.add_argument('--content_field', type=str,default='Content', help='field to use as text')
    parser.add_argument('--vdb_name', type=str,default='vdb', help='name of vdb')
    parser.add_argument('--num_processes', type=int, default=4, help='number of processes to use')
    parser.add_argument('--base_rwkv_model', type=str, default='/media/yueyulin/bigdata/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth', help='base rwkv model to use')
    parser.add_argument('--lora_path', type=str, default='/media/yueyulin/KINGSTON/models/rwkv6/lora/bi-encoder/add_mlp_in_batch_neg/epoch_0_step_200000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth', help='lora model to use')
    parser.add_argument('--is_qdrant', action='store_true', default=False,help='use qdrant instead of chroma')
    parser.add_argument('--bge_path',type=str,default='/media/yueyulin/KINGSTON/models/bge-m3')
    parser.add_argument('--use_bge',action='store_true',default=False,help='use bge instead of rwkv')
    parser.add_argument('--need_clean',action='store_true',default=False,help='clean the input file before adding to db')
    parser.add_argument('--host',type=str,default='localhost',help='host of qdrant')
    args = parser.parse_args()
    is_dir = True
    if os.path.isfile(args.input):    
        is_dir = False
    
    ###chroma run --path db_path
    #run subprocess chroma run --path /db_path
    if not args.is_qdrant:
        os.makedirs(args.output, exist_ok=True)
        db_path = os.path.join(args.output, args.vdb_name)
        import subprocess
        db_proc = subprocess.Popen(['chroma', 'run', '--path', db_path])

    if is_dir:
        with mp.Pool(args.num_processes) as pool:
            for file in os.listdir(args.input):
                if file.endswith('.json') or file.endswith('.jsonl'):
                    pool.apply_async(add_record_to_db,args = (os.path.join(args.input,file),args.id_field,args.content_field,args.base_rwkv_model,args.lora_path,args.is_qdrant,args.use_bge,args.bge_path,args.need_clean,args.host))
            pool.close()
            pool.join()
    else:
        add_record_to_db(args.input,args.id_field,args.content_field,args.base_rwkv_model,args.lora_path,args.is_qdrant,args.use_bge,args.bge_path,args.need_clean,args.host)
    if not args.is_qdrant:
        print(colorama.Fore.YELLOW+f"closing db {db_path}"+colorama.Fore.RESET)
        db_proc.terminate()
        db_proc.wait()
        print(colorama.Fore.GREEN+f"finished building vdb {db_path}"+colorama.Fore.RESET)