# 04c_hybrid_practice.py —— 动手实践：给 04 加上混合检索（填空版）
# 规则：把每个 "===== 填空 N =====" 下面的占位行替换成你的代码（一行或多行）
# 说明：
#   1. 骨架 = 04 的多模态检索 + 01 的混合检索；每个空都能在
#      work_hybrid_multimodal_search.py 或 01_hybrid_search.py 里找到同款
#   2. 三条向量路：multimodal(768, HNSW+COSINE) / text_dense(1024, AUTOINDEX+IP)
#      / text_sparse(SPARSE_INVERTED_INDEX+IP)
#   3. 运行（必须在 code/C3 目录下）：
#      HF_HUB_OFFLINE=1 python 04c_hybrid_practice.py
#   4. 验证标准：输出应和 work_hybrid_multimodal_search.py 的四模式对照一致
#      （白龙四路全第一；HYBRID 分数约 0.0328 / 0.0320 / 0.0161）

import json
import os
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import (connections, MilvusClient, FieldSchema, CollectionSchema,
                      DataType, Collection, AnnSearchRequest, RRFRanker)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# ========== 第 1 步：常量与连接（直接给你） ==========
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3/dragon"
METADATA_PATH = "../../data/C4/metadata/dragon.json"
COLLECTION_NAME = "practice_hybrid_demo"
MILVUS_URI = "http://localhost:19530"

connections.connect(uri=MILVUS_URI)
milvus_client = MilvusClient(uri=MILVUS_URI)

# ========== 第 2 步：两个编码器 ==========
visual_model = Visualized_BGE(model_name_bge=MODEL_NAME, model_weight=MODEL_PATH)
visual_model.eval()

# ===== 填空 1 =====  # BGE-M3 编码器：use_fp16=False, device="cpu"（01 第 18 行同款）
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")

def encode_multimodal(image_path, text):
    """图+文融合向量（04 的老朋友，直接给你）"""
    with torch.no_grad():
        emb = visual_model.encode(image=image_path, text=text)
    return emb.tolist()[0]

# ========== 第 3 步：读 dragon.json（直接给你） ==========
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

docs = []  # 每条龙拼接后的文本（向量化原料）
for item in dataset:
    parts = [item.get('title', ''), item.get('description', ''),
             item.get('location', ''), item.get('environment', '')]
    docs.append(' '.join(filter(None, parts)))

# ========== 第 4 步：建 Collection（三向量字段） ==========
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)

multimodal_dim = len(encode_multimodal(dataset[0]['path'], docs[0]))
dense_dim = ef.dim["dense"]

fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="multimodal_vector", dtype=DataType.FLOAT_VECTOR, dim=multimodal_dim),
    # ===== 填空 2 =====  # 文本稀疏向量字段：SPARSE_FLOAT_VECTOR
    FieldSchema(name='text_sparse_vector', dtype=DataType.SPARSE_FLOAT_VECTOR),
    # ===== 填空 3 =====  # 文本密集向量字段：FLOAT_VECTOR，dim 用 dense_dim
    FieldSchema(name='text_dense_vector', dtype=DataType.FLOAT_VECTOR, dim=dense_dim)
]
collection = Collection(name=COLLECTION_NAME,
                        schema=CollectionSchema(fields, description="04+hybrid 动手实践"),
                        consistency_level="Strong")

# ========== 第 5 步：三套索引 ==========
multimodal_index = {"index_type": "HNSW", "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 256}}
collection.create_index("multimodal_vector", multimodal_index)

# ===== 填空 4 =====  # 稀疏索引：SPARSE_INVERTED_INDEX + IP（01 第 50 行同款）
collection.create_index("text_sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})

# ===== 填空 5 =====  # 密集索引：AUTOINDEX + IP（01 第 54 行同款）
collection.create_index("text_dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})

collection.load()

# ========== 第 6 步：插入（每条龙编码三次） ==========
paths, titles, descriptions = [], [], []
multimodal_vecs, sparse_vecs, dense_vecs = [], [], []

for img_data, text_content in zip(dataset, docs):
    paths.append(img_data['path'])
    titles.append(img_data['title'])
    descriptions.append(img_data['description'])
    multimodal_vecs.append(encode_multimodal(img_data['path'], text_content))

    # ===== 填空 6 =====  # BGE-M3 编码文本：ef([text_content])，稀疏取 ._getrow(0) 追加进
    # sparse_vecs，密集取 [0] 追加进 dense_vecs（work_hybrid 第 243-244 行同款）
    text_embeddings = ef([text_content])
    sparse_vecs.append(text_embeddings['sparse']._getrow(0))
    dense_vecs.append(text_embeddings['dense'][0])

collection.insert([paths, titles, descriptions, multimodal_vecs, sparse_vecs, dense_vecs])
collection.flush()
print(f"插入完成，总数: {collection.num_entities}")

# ========== 第 7 步：四模式搜索 ==========
query_image = os.path.join(DATA_DIR, "query.png")
query_text = "悬崖上的巨龙"
output_fields = ["image_path", "title", "description"]

search_params = {"metric_type": "IP", "params": {}}                # 文本两路用 IP
cosine_params = {"metric_type": "COSINE", "params": {"ef": 128}}   # 多模态路用 COSINE

def show(title, results):
    print(f"\n--- [{title}] ---")
    for i, hit in enumerate(results):
        print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")

# [1] 多模态（图+文）—— 04 老路线，直接给你
mm_vec = encode_multimodal(query_image, query_text)
mm_results = collection.search([mm_vec], "multimodal_vector",
                               param=cosine_params, limit=3, output_fields=output_fields)[0]
show("MULTIMODAL", mm_results)

# ===== 填空 7 =====  # 编码查询文本：ef([query_text])，取出 dense_vec 和 sparse_vec
# （01 第 131-133 行同款）
query_embeddings = ef([query_text])
dense_vec = query_embeddings["dense"][0]
sparse_vec = query_embeddings["sparse"]._getrow(0)

# ===== 填空 8 =====  # 密集向量单搜：anns_field="text_dense_vector"，param=search_params，
# limit=3，output_fields=output_fields（01 第 154-161 行同款，不带 expr）
dense_results = collection.search(
    [dense_vec],
    anns_field="text_dense_vector",
    param=search_params,
    limit=3,
    output_fields=output_fields
)[0]
show("DENSE", dense_results)

# ===== 填空 9 =====  # 稀疏向量单搜：同上，但 anns_field="text_sparse_vector"
sparse_results = collection.search(
    [sparse_vec],
    anns_field="text_sparse_vector",
    param=search_params,
    limit=3,
    output_fields=output_fields
)[0]
show("SPARSE", sparse_results)

# ===== 填空 10 =====  # 三件套之一：RRF 融合器，k=60（01 第 185 行同款）
rerank = RRFRanker(k=60)

# ===== 填空 11 =====  # 三件套之二：两个 AnnSearchRequest（密集请求 + 稀疏请求，limit=3）
# （01 第 188-189 行同款）
dense_req = AnnSearchRequest([dense_vec], "text_dense_vector", search_params, limit=3)
sparse_req = AnnSearchRequest([sparse_vec], "text_sparse_vector", search_params, limit=3)

# ===== 填空 12 =====  # 三件套之三：hybrid_search 执行融合（01 第 192-197 行同款）
hybrid_results = collection.hybrid_search(
    [sparse_req, dense_req],
    rerank=rerank,
    limit=3,
    output_fields=output_fields
)[0]
show("HYBRID", hybrid_results)

# ========== 第 8 步：清场（直接给你） ==========
collection.release()
milvus_client.drop_collection(COLLECTION_NAME)
print("\n已清场。")
