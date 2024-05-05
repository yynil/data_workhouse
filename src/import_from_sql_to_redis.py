import os
import json
import argparse
from sqlite_utilities import SqliteDictWrapper
import redis
from multiprocessing import Pool
def import_db_to_redis(db_file, redis_host, redis_port, redis_db=0):
    # Initialize Redis client
    try:
        redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        print("Connected to Redis")
    except redis.exceptions.ConnectionError:
        print("Failed to connect to Redis")
        return
    from tqdm import tqdm
    db = SqliteDictWrapper(db_file)
    progress_bar = tqdm(len(db), desc=f'Importing keys for {db_file} to Redis')
    batch_size = 4096
    with redis_client.pipeline() as pipe:
        for i,uuid in enumerate(db):
            value = db[str(uuid)]
            document = value['document']

            try:
                pipe.set(uuid, document)
            except TypeError as e:
                print(f"Failed to serialize document for UUID {uuid}: {e}")
            if i % batch_size == 0:
                pipe.execute()
                progress_bar.update(batch_size)
        pipe.execute()
    db.close()
    print("Import to Redis complete")

def main():
    parser = argparse.ArgumentParser(description="Import data from SQLite to Redis.")
    parser.add_argument("--db_file", required=True, help="Path to the SQLite database file or directory containing .db files")
    parser.add_argument("--redis-host", default="localhost", help="Redis server host (default: localhost)")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis server port (default: 6379)")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database number (default: 0)")
    parser.add_argument("--num-processes", type=int, default=4, help="Number of processes for parallel import when importing from a directory (default: 4)")

    args = parser.parse_args()

    if os.path.isdir(args.db_file):
        with Pool(args.num_processes) as pool:
            for file in os.listdir(args.db_file):
                db_path = os.path.join(args.db_file, file)
                if db_path.endswith('.db') and os.path.isfile(db_path):
                    pool.apply_async(import_db_to_redis, args=(db_path, args.redis_host, args.redis_port, args.redis_db))
            pool.close()
            pool.join()
    else:
        # If it's a single file, import directly
        if os.path.isfile(args.db_file):
            import_db_to_redis(args.db_file, args.redis_host, args.redis_port, args.redis_db)
        else:
            print(f"{args.db_file} is neither a valid file nor a directory.")

if __name__ == "__main__":
    main()