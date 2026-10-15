import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

# 使用 AIHubmix
# Settings.llm = OpenAILike(
#     model="glm-4.7-flash-free",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://aihubmix.com/v1",
#     is_chat_model=True
# )

#使用 DeepSeek
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com",
    is_chat_model=True,
    temperature=1.5
)
#可以手动设置切块的参数
# Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
#中文嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

#数据加载，加载markdown文件
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

#一站式建索引：内部自动完成 切块 + 向量化 + 建库
index = VectorStoreIndex.from_documents(docs)

#创建问答引擎，检索和打包生成好的提问入口
query_engine = index.as_query_engine()

#打印框架内置的提示词模板
print(query_engine.get_prompts())

#提问：检索→填模板→调LLM→返回答案
print(query_engine.query("文中举了哪些例子?"))