import os
import sys
parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rwkv_ext_dir = os.path.join(parent_parent_dir, 'RWKV_LM_EXT')
sys.path.append(rwkv_ext_dir)
print(f'added {rwkv_ext_dir} to sys.path')
import chromadb.db
import colorama
import sqlite3
import argparse
import chromadb
import orjson
import tqdm
from colorama import Fore, Style
from datetime import datetime

def add_record_to_db(input_file,output_dir, id_field, content_field,rwkv_base,lora_path,use_bge=False,bge_path=None,need_clean=False,host='localhost',batch_size=8,is_blob=False):
    import uuid
    import os
    if not is_blob:
        try:
            db_file = os.path.join(output_dir, os.path.basename(input_file).split('.')[0] + '.db')
            print(Fore.YELLOW + f'Saving data to {db_file}' + Style.RESET_ALL)
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    UUID TEXT PRIMARY KEY,
                    Embedding TEXT,
                    Document TEXT
                )
            """)
            print(colorama.Fore.GREEN + f'Database file {db_file} created' + Style.RESET_ALL)
            #select count(*) from embeddings as skipped_lines
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            skipped_lines = cursor.fetchone()[0]
        except:
            print(colorama.Fore.RED + f'Error creating database file {db_file}' + Style.RESET_ALL)
    else:
        from sqlite_utilities import SqliteDictWrapper
        db_file = os.path.join(output_dir, os.path.basename(input_file).split('.')[0] + '.db')
        print(Fore.YELLOW + f'Saving data to {db_file}' + Style.RESET_ALL)
        wrapper = SqliteDictWrapper(db_file)
        print(colorama.Fore.GREEN + f'Database file {db_file} created' + Style.RESET_ALL)
        skipped_lines = len(wrapper)
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
            try:

                from FlagEmbedding import BGEM3FlagModel
                encoder = BGEM3FlagModel(bge_path)
            except:
                import traceback
                traceback.print_exc()
                print(colorama.Fore.RED + f"bge_path {bge_path} not found" + colorama.Style.RESET_ALL)
    print(colorama.Fore.GREEN + f"adding records from {input_file}" + colorama.Style.RESET_ALL)
    batch_insert = 1000
    line_counter=0
    index = skipped_lines
    print(colorama.Fore.GREEN + f"skipped {skipped_lines} lines" + colorama.Style.RESET_ALL)
    with open(input_file, 'r', encoding='UTF-8') as f:
        meta_file = os.path.join(os.path.dirname(input_file), os.path.basename(input_file).split('.')[0] + '.txt')
        lines = f.readlines()
        progress_bar = tqdm.tqdm(lines[skipped_lines:],desc=f'adding {input_file} to vdb')
        documents=[]
        all_embeddings=[]
        for line in progress_bar:
            try:
                data_obj = orjson.loads(line)
                content = data_obj[content_field]
            except:
                import traceback
                traceback.print_exc()
                print(colorama.Fore.RED + f"failed to get content {line}" + colorama.Style.RESET_ALL)
                continue
            try:        
                if need_clean:
                    content = clean_in_order(content,clean_functions)
                if use_bge:
                    embeddings = encoder.encode([content],batch_size=1,max_length=2048)['dense_vecs'].tolist()
                else:
                    embeddings = encoder.encode_texts([content]).tolist()
            except:
                import traceback
                traceback.print_exc()
                print(colorama.Fore.RED + f"failed to encode content {content}" + colorama.Style.RESET_ALL)
            try:
                documents.append(content)
                all_embeddings.extend(embeddings)
            except:
                print(colorama.Fore.RED + f"failed to append content" + colorama.Style.RESET_ALL)   
            import uuid
            if len(documents) >= batch_insert:
                    print(colorama.Fore.YELLOW+f'adding {batch_insert} records to vdb'+colorama.Style.RESET_ALL)
                    uuids = [str(uuid.uuid4()) for i in range(len(documents))]
                    now = datetime.now()
                    for uuid, embedding, document in zip(uuids, all_embeddings, documents):
                        if not is_blob:
                            try:
                                cursor.execute("INSERT INTO embeddings (UUID, Embedding, Document) VALUES (?, ?, ?)", (uuid, str(embedding), document))
                                conn.commit()
                            except:
                                print(colorama.Fore.RED + f"failed to insert record" + colorama.Style.RESET_ALL)
                            try:
                                with open(meta_file, 'a') as meta_file_handle:
                                    meta_file_handle.write(f'{line_counter}\n') 
                            except:
                                print(colorama.Fore.RED + f"failed to write to meta file" + colorama.Style.RESET_ALL)  
                        else:
                            wrapper[index] = {'uuid':uuid,'embedding':embedding,'document':document}              
                            index += 1
                    if is_blob:
                        wrapper.commit()
                    
                    elapsed = datetime.now() - now
                    print(colorama.Fore.YELLOW+f'added {batch_insert} records in {elapsed}'+colorama.Style.RESET_ALL)

                    documents=[]
                    all_embeddings=[]
           
            line_counter +=1
        import uuid
        if len(documents) > 0:
            uuids = [str(uuid.uuid4()) for i in range(len(documents))]
            for uuid, embedding, document in zip(uuids, all_embeddings, documents):
                if not is_blob:
                    try:
                        cursor.execute("INSERT INTO embeddings (UUID, Embedding, Document) VALUES (?, ?, ?)", (uuid, str(embedding), document))
                        conn.commit()
                    except:
                        print(colorama.Fore.RED + f"failed to insert record" + colorama.Style.RESET_ALL)
                    try:
                        with open(meta_file, 'a') as meta_file_handle:
                            meta_file_handle.write(f'{line_counter}\n') 
                    except:
                        print(colorama.Fore.RED + f"failed to write to meta file" + colorama.Style.RESET_ALL)  
                else:
                    wrapper[uuid] = {'embedding':embedding,'document':document}              
                    index += 1
            if is_blob:
                wrapper.commit()
            else:
                conn.commit()
                cursor.close()
                conn.close()
                with open(meta_file, 'a') as meta_file_handle:
                    meta_file_handle.write(f'{line_counter}\n')

   
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build vdb from corpus')
    parser.add_argument('--input', type=str, help='input file or directory')
    parser.add_argument('--output', type=str, help='output directory')
    parser.add_argument('--id_field', type=str,default='ID', help='field to use as id')
    parser.add_argument('--content_field', type=str,default='Content', help='field to use as text')
    parser.add_argument('--meta_file', type=str, default='meta.txt', help='Path to the meta file')
    parser.add_argument('--vdb_name', type=str,default='vdb', help='name of vdb')
    parser.add_argument('--num_processes', type=int, default=4, help='number of processes to use')
    parser.add_argument('--base_rwkv_model', type=str, default='/media/yueyulin/bigdata/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth', help='base rwkv model to use')
    parser.add_argument('--lora_path', type=str, default='/media/yueyulin/KINGSTON/models/rwkv6/lora/bi-encoder/add_mlp_in_batch_neg/epoch_0_step_200000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth', help='lora model to use')
    parser.add_argument('--bge_path',type=str,default='/media/yueyulin/KINGSTON/models/bge-m3')
    parser.add_argument('--use_bge',action='store_true',default=False,help='use bge instead of rwkv')
    parser.add_argument('--need_clean',action='store_true',default=False,help='clean the input file before adding to db')
    parser.add_argument('--batch_size',type=int,default=32,help='batch size for encoding')
    parser.add_argument('--host',type=str,default='localhost',help='host of qdrant')
    parser.add_argument('--is_blob',action='store_true',default=False,help='If the embedding is stored as BLOB')
    args = parser.parse_args()
    is_dir = True
    if os.path.isfile(args.input):    
        is_dir = False
    
    ###chroma run --path db_path
    #run subprocess chroma run --path /db_path


    if is_dir:
        import multiprocessing as mp
        with mp.Pool(args.num_processes) as pool:
            for file in os.listdir(args.input):
                if file.endswith('.json') or file.endswith('.jsonl'):
                    pool.apply_async(add_record_to_db,args = (os.path.join(args.input,file),args.output,args.id_field,args.content_field,args.base_rwkv_model,args.lora_path,args.use_bge,args.bge_path,args.need_clean,args.host, args.batch_size,args.is_blob))
            pool.close()
            pool.join()
    else:
        add_record_to_db(args.input,args.output,args.id_field,args.content_field,args.base_rwkv_model,args.lora_path,args.use_bge,args.bge_path,args.need_clean,args.host, args.batch_size,args.is_blob)