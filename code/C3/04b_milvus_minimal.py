# 04b_milvus_minimal.py —— 最小版 Milvus 五步曲（背默练习）
# 规则：把每个 "===== 填空 N =====" 替换成你的代码（一行或多行）
# 提示：骨架就是那 7 个动词；字段怎么定义，回忆 04 第 4 段的 FieldSchema
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

COLLECTION_NAME = "hello_milvus"

# ========== 第 1 步：拿起对讲机 ==========
# ===== 填空 1 =====  # 创建 Milvus 客户端，连 http://localhost:19530
milvus_client = MilvusClient(uri="http://localhost:19530") 
# ========== 第 2 步：建图书馆（先拆旧的，再建新的） ==========
# ===== 填空 2 =====  # 如果 COLLECTION_NAME 已存在，先 drop 掉（幂等）
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(collection_name=COLLECTION_NAME)
fields = [
    # ===== 填空 3 =====  # 字段1：id，INT64，主键，自增
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    # ===== 填空 4 =====  # 字段2：vector，FLOAT_VECTOR，dim=4
    FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=4),
    # ===== 填空 5 =====  # 字段3：text，VARCHAR，max_length=64
    FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=64)
]
schema = CollectionSchema(fields, description="最小练习")
# ===== 填空 6 =====  # 用上面的 schema 创建 Collection
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema = schema)
print('Schema结构:',schema)
# ========== 第 3 步：插 3 本书（数据给你，插入你来） ==========
data = [
    {"vector": [1, 0, 0, 0], "text": "苹果"},
    {"vector": [0, 1, 0, 0], "text": "香蕉"},
    {"vector": [0, 0, 1, 0], "text": "橙子"},
]
# ===== 填空 7 =====  # 批量插入，并打印 result['insert_count']
data_to_insert = []
for d in data:
    data_to_insert.append({"vector": d["vector"], "text": d["text"]})

if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")

# ========== 第 4 步：建检索卡 + 搬上阅览桌 ==========
# ===== 填空 8 =====  # prepare_index_params 后 add_index：field_name="vector"，HNSW，COSINE
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name = 'vector',
    index_type = 'HNSW',
    metric_type = 'COSINE',
    params = {"M": 16, "efConstruction": 256}
)
# ===== 填空 9 =====  # create_index
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
# ===== 填空 10 =====  # load_collection
milvus_client.load_collection(collection_name=COLLECTION_NAME)
# ========== 第 5 步：检索 ==========
query_vector = [1, 0, 0, 0]
# ===== 填空 11 =====  # search：data=[query_vector]，output_fields=["text"]，limit=3，COSINE+ef
search_results = milvus_client.search(
    collection_name = COLLECTION_NAME,
    data = [query_vector],
    output_fields = ['text'],
    limit = 3,
    search_params = {"metric_type":'COSINE', "params": {'ef': 128}}
)[0]
# 变量名必须是 search_results，下面打印代码给你，不动：
for i, hit in enumerate(search_results):
    print(f"Top {i+1}: text='{hit['entity']['text']}', 距离={hit['distance']:.4f}")

# ========== 第 6 步：清场 ==========
# ===== 填空 12 =====  # release_collection
milvus_client.release_collection(collection_name = COLLECTION_NAME)
# ===== 填空 13 =====  # drop_collection
milvus_client.drop_collection(collection_name = COLLECTION_NAME)
print("已清场。")
