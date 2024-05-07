import redis

def get_redis_client(redis_host, redis_port, redis_db=0):
    return redis.Redis(host=redis_host, port=redis_port, db=redis_db)

def get_doc_from_redis(redis_client,uuid):
    bs = redis_client.get(uuid)
    return str(bs,encoding='utf-8') if bs is not None else ''

def get_docs_from_redis(redis_client,uuids):
    return [str(bs,encoding='utf-8') if bs is not None else '' for bs in redis_client.mget(uuids)]

if __name__ == '__main__':
    redis_host = '192.168.1.36'
    redis_port = 6379
    redis_db = 0
    redis_client = get_redis_client(redis_host, redis_port, redis_db)
    uuids = ['2f319d9c-2f78-4541-9dd7-f3c810618a80','efd9ff26-8f38-4e31-bc3b-fe5629a65fde','8057932a-e4df-4a8a-aa6f-fe9cb8c5ab04','c15d8fd7-d27b-4c09-82cf-2e3d3388bbc5','s','f895a683-77fe-46c1-a082-0dbd8f04e7df','34a3eddc-0535-4e9a-b25c-d48a920c7cec']
    docs = get_docs_from_redis(redis_client,uuids)
    for doc in docs:
        print(doc)
        print('-------------------')