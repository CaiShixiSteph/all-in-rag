# ============================================
# C1 闭卷考试：凭理解补全 RAG 五步
# 规则：
#   1. 先关掉 01_langchain_example.py，不许看（它就是答案卷）
#   2. 每个 ＿＿(N)＿＿ 填一行代码或一个表达式
#   3. 卡住超过 10 分钟就把 N 记下来跳过，最后一起发我
#   4. 填完能正常运行、输出答案 = 过关
# ============================================
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# ---------- 第1步：加载 ----------
# 提示：先创建加载器（告诉它文件在哪），再调用 load() 读出文档列表
markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"
loader = UnstructuredMarkdownLoader(markdown_path)        # 创建加载器
docs = loader.load()        # 读取文件，结果存进 docs

# ---------- 第2步：切块 ----------
# 提示：先创建切块器，再切；切完顺手打印块数量看看效果
text_splitter = RecursiveCharacterTextSplitter()        # 创建切块器
chunks = text_splitter.split_documents(docs)        # 把 docs 切成块，存进 chunks

# ---------- 第3步：向量化 + 建库 ----------
# 提示：先建向量库（把翻译员 embeddings 绑进去），再把块翻译成向量上架
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = InMemoryVectorStore(embeddings)        # 建向量库
vectorstore.add_documents(chunks)        # 把块翻译成向量并上架

# ---------- 第4步：检索 ----------
question = "文中举了哪些例子？"
retrieved_docs = vectorstore.similarity_search(question, k=3)       # 检索与问题最像的 3 块，结果存进 retrieved_docs
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)       # 把 3 块原文拼成一个字符串，存进 docs_content
                 # （提示：for doc in ... 取 doc.page_content，用 "\n\n".join 拼接）

# ---------- 第5步：生成 ----------
# 提示：模板必须包含——
#   ① 三条规则：只根据上下文回答 / 完全基于上下文 / 没有信息就说"抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"
#   ② 两个空位 {context} 和 {question}
#   ③ 结尾是"回答:"（留白让模型续写）
prompt = ChatPromptTemplate.from_template("""请你根据下面的上下文信息来回答问题，并确保你的回答完全基于上下文信息。
如果上下文中没有足够的信息来回答问题，请直接表达：“抱歉，我无法根据提供的上下文信息来回答问题。”
question: {question}
context: {context}
回答:"""                                         
                                          )

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY"),   # 填环境变量名（你亲手创建过它）
    base_url="https://api.deepseek.com"              # 填 DeepSeek 官方地址
)

       # 把 question 和 context 填进模板，结果存进 full_prompt
answer = llm.invoke(prompt.format(question = question, context = docs_content))       # 调用大模型生成答案，结果存进 answer
print(answer)
