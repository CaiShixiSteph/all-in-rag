import os
os.environ['HF_HUB_OFFLINE'] = '1'          # 模型已在本地缓存，禁止联网检查
os.environ['TRANSFORMERS_OFFLINE'] = '1'    # transformers 也走离线
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding('BAAI/bge-small-zh-v1.5')

# 2. 创建索引
store_context = StorageContext.from_defaults(persist_dir="./llamaindex_index_store")
index = load_index_from_storage(store_context)
print("LlamaIndex 索引已加载成功。")

# 3. 检索器
retriever = index.as_retriever(similarity_top_k=1)

# 4. 执行查询并打印结果
question = "LlamaIndex是做什么的？"
nodes = retriever.retrieve(question)
print(f"\n查询: '{question}'")
for n in nodes:
    print(f"相似度得分：{n.score:.4f}, 文档内容: {n.node.get_content()}")
