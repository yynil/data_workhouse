if __name__ == '__main__':
    from qdrant_client import QdrantClient
    client = QdrantClient("localhost")
    print('Client created')
    client.close()
    