import argparse
import os
import shutil
from sqlite_utilities import SqliteDictWrapper

def convert(db_path,output_dir):
    if not os.path.exists(db_path):
        print(f"File {db_path} does not exist")
        return

    output_db_file = os.path.join(output_dir,os.path.basename(db_path))

    try:

        # Open the db file
        db = SqliteDictWrapper(db_path)
        out_db = SqliteDictWrapper(output_db_file)
        from tqdm import tqdm
        progress_bar = tqdm(len(db), desc=f'Converting keys for {db_path} to {output_db_file}', unit='keys', unit_scale=True, unit_divisor=1024)
        batch_commit = 4096
        for key in db:
            progress_bar.update(1)
            value = db[key]
            # {'uuid':uuid,'embedding':embedding,'document':document}
            uuid = value['uuid']
            embedding = value['embedding']
            document = value['document']
            out_db[uuid] = {'embedding':embedding,'document':document}
            if len(out_db) % batch_commit == 0:
                out_db.commit()
        # Close the db file
        out_db.commit()
        out_db.close()
        db.close()
    except Exception as e:
        print(f"Error converting {db_path} to {output_db_file}")
        print(e)

parser = argparse.ArgumentParser(description='Convert a db key from int to string')
parser.add_argument('--db_path', type=str, help='Path to the db file')
parser.add_argument('--output_dir', type=str, help='Path to the output directory')
parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use')
args = parser.parse_args()

if os.path.isfile(args.db_path):
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    convert(args.db_path,args.output_dir)
else:
    from multiprocessing import Pool
    with Pool(args.num_processes) as pool:
        for file in os.listdir(args.db_path):
            db_path = os.path.join(args.db_path,file)
            if db_path.endswith('.db') and os.path.isfile(db_path):
                os.makedirs(args.output_dir,exist_ok=True)
                pool.apply_async(convert,args=(db_path,args.output_dir))
        pool.close()
        pool.join()