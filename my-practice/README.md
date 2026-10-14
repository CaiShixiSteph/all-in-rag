# 我的练习记录

## 2026-10-14 跑通 C1 RAG 示例
- RAG 五步：加载 → 切块 → 向量化 → 检索 → 生成
- 踩坑记录：
  -nltk数据被墙 → 用ghfast.top代理下载（见scripts/fix_nltk_mirror.py）
  - embedding模型走HF镜像（代码第3行开关）
  - 脚本要用相对路径，必须在code/C1目录下运行
  - 新建环境变量后要重启VSCode才生效
- 输出：模型基于检索到的3个切片回答了"文中举了哪些例子"，共13个例子
尝试替换了不相干的问题，结果成功输出“抱歉…”，还替换了另一个问题——什么是强化学习，获得了不同的正确的输出
另外还尝试修改了k值，从3改为5后消耗token变多了

## 2026-10-14 C1 代码问答笔记

**1. 为什么需要 loader？**
loader 负责：格式解析（PDF/Word/网页）、结构识别（标题/段落）、
loader = 干活的工人（构造时只是"请人"，文件还没读）；docs = 工人交出的产品（load() 的返回值）。

**2. 切块（chunking）**
- 为什么切：嵌入模型输入长度限制、检索粒度、上下文窗口限制
- 递归分割：按 \n\n段落 → \n行 → 空格 → 字符 逐级尝试，尽量在语义边界切
- chunk_size=4000（默认，字符数）、chunk_overlap=200（相邻块重叠，防关键句被切断）
这里是将上面读取到的docs切成块

**3. 向量库两行**
- InMemoryVectorStore(embeddings)：建空仓库+绑翻译员（检索时问题必须用同一个模型嵌入才能比）
- add_documents(chunks)：内部把每块翻译成向量，(向量+原文)成对存
- 判读技巧：有等号左边=取出（load/search/invoke），无等号=存入（add_documents）
形象的比喻：
1. 第1行：你开了一家档案室 vectorstore，并雇了翻译员 embeddings。此时档案室空的，一个档案都没有
2. 第2行：你把一叠稿件 chunks 送进档案室。翻译员在内部把它们翻译成向量、归档（第2行括号里没写 embeddings，但翻译照样发生——因为翻译员第1行就上岗了

**5. 生成器表达式**
doc.page_content for doc in retrieved_docs = "对列表里每个 Document 取正文"，
等价于 for 循环+append；doc是循环变量名
