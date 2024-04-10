# 文本去重设计和计划

## 文本去重目标

- 清理文本特殊字符
- 长文本分割
- 文本去重

## 开发方案和计划

| 内容 | 方案 | 计划完成时间 |  状态 |
| ---- | ---- | ------------ | ---|
| 清理文本特殊字符 | Unstructed IO Clean接口 | 2022-04-09 |Done
| 文本去重 | Embeddings model for short texts| 2022-04-13| 50%
| 去重Pipeplien| File in File Out | 2022-04-15| On Schedule

## VDB store

### ChromaDB
 It's an embedded vectordb and suitable for testing and prototype

### qdrant
It's a massive production ready vectordb with high performance.

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

Create a collection to insert data
```bash
curl -X PUT -H "Content-Type: application/json" -d '{
    "vectors": {
      "size": 1024,
      "distance": "Cosine"
    }
}' http://localhost:6333/collections/mycorpus_vdb
```

## Create vdb
Using bge-m3 because we need to encode English as well as Chinese.

- Download BAAI/bge-m3 model locally. Follow the instructions: https://hf-mirror.com/ 
- Install FlagEmbedding : pip install -U FlagEmbedding
```python
python src/02_vdb_build.py --input /home/yueyulin/tmp/RedPajamaCommonCrawl --is_qdrant --use_bge --bge_path /media/yueyulin/KINGSTON/models/bge-m3 --need_clean --content_field text --num_processes 8
```
This script is to use 8 processes to handle files in input path and write to qdrant vector database.