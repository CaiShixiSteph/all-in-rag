# 我的练习记录

## 2026-10-14 跑通 C1 RAG 示例
- RAG 五步：加载 → 切块 → 向量化 → 检索 → 生成
- 踩坑记录：
  -nltk数据被墙 → 用ghfast.top代理下载（见scripts/fix_nltk_mirror.py）
  - embedding模型走HF镜像（代码第3行）
  - 脚本要用相对路径，必须在code/C1目录下运行
  - 新建环境变量后要重启VSCode才生效
- 输出：模型基于检索到的3个切片回答了"文中举了哪些例子"，共13个例子
尝试替换了不相干的问题，结果成功输出“抱歉…”，还替换了另一个问题——“什么是强化学习”，获得了不同的正确的输出
另外还尝试修改了k值，从3改为5后消耗token变多了

## 2026-10-14 C1 代码问答笔记

**1. 为什么需要loader？**
loader负责：格式解析（PDF/Word/网页）、结构识别（标题/段落）、
loader就相当于干活的工人（构造时只是"请人"，文件还没读）；docs就是工人交出的产品（load() 的返回值）。

**2. 切块（chunking）**
- 为什么切：嵌入模型输入长度限制、检索粒度、上下文窗口限制
- 递归分割：按 \n\n段落 → \n行 → 空格 → 字符 逐级尝试，尽量在语义边界切
- chunk_size=4000（默认，字符数）、chunk_overlap=200（相邻块重叠，防关键句被切断）
这里是将上面读取到的docs切成块

**3. 向量库两行**
- InMemoryVectorStore(embeddings)：建空仓库+绑翻译员（检索时问题必须用同一个模型嵌入才能比）
- add_documents(chunks)：内部把每块翻译成向量，(向量+原文)成对存
形象的比喻：
1. 第1行：你开了一家档案室 vectorstore，并雇了翻译员 embeddings。此时档案室空的，一个档案都没有
2. 第2行：你把一叠稿件 chunks 送进档案室。翻译员在内部把它们翻译成向量、归档（第2行括号里没写 embeddings，但翻译照样发生——因为翻译员第1行就上岗了

**5. 生成器表达式**
doc.page_content for doc in retrieved_docs = "对列表里每个 Document 取正文"，
等价于 for 循环+append；doc是循环变量名

## 2026-10-15 C1第二个示例：LlamaIndex 版 RAG 跑通

**核心收获：同样的RAG五步，看到了两种框架风格**
- LangChain可以看作自己进后厨：切块/建库/检索/拼接每步显式可见
- LlamaIndex可以看作点外卖：VectorStoreIndex.from_documents()一行完成 切块+向量化+建库；
  query()一行完成 检索+拼接+生成。中间步骤全藏在框架里

**新概念**
- Settings：全局注册中心。登记好LLM和嵌入模型后，所有组件自动使用，不用每次传参
- OpenAILike：万能接口。DeepSeek的API是OpenAI兼容格式，api_base指向谁就调谁
- query_engine：问答引擎，检索+生成打包好的"提问入口"
- get_prompts()：查看框架内置的默认prompt模板，不同于LangChain版要自己写，这版是现成的

**踩坑记录**
- nltk缺stopwords语料（llama_index 的切块器也依赖 nltk）→ gh-proxy.com 代理下载解决
- OpenAILike默认用老式completions接口，DeepSeek只支持chat接口 → 报 400
这里我们加了一行 is_chat_model=True 就可以解决。

**实验观察**
- 同样问"文中举了哪些例子？"：LangChain 版答了13个例子，LlamaIndex 版只答4个
- 原因：内置prompt模板不同，以及默认检索块数不同（LlamaIndex 默认top2）
- 结论：RAG骨架相同，但每一环的默认选择不同 → 答案质量不同。
  调参（块大小、检索数量、prompt 写法）就是 RAG 工程师的核心工作，接下来我做了一系列调参实验
方法：固定问题"文中举了哪些例子？"，逐一单变量实验：
基准：默认参数（top_k=2, chunk_size=1024），并且有4个例子：餐馆/广告/挖油/游戏
实验1：top_k 2→5，仍输出了4个例子，但内容全换，变成了羚羊/股票/K-臂赌博机等核心概念。
实验2：chunk_size 1024→512，答案一字未变（块数实际翻倍） |
实验3：temperature=1.5 ，仍输出4个例子，仅措辞微变（"看看效果"/"大招"多空格），说明影响的是大语言模型输出的自由度。

结论：
- similarity_top_k是粗调，改它就导致检索材料变，答案内容立刻变
- chunk_size是微调，我猜测它对"覆盖型"简单问题无感，可能得到长文档/跨段问题才发力
- temperature只管"怎么说"，内容被检索上下文钉死，只影响措辞自由度；且它默认很低，答案接近确定性
- 答案条数由LLM决定，不由检索参数决定

## C1动手练习记录（LangChain 版）

**练习1：过滤输出参数**
- llm.invoke()返回的是AIMessage对象，print整个对象会带出content、response_metadata、usage_metadata等字段
- 只取正文的方法是：print(answer.content)

**练习2：chunk_size / chunk_overlap实验**（问题固定"文中举了哪些例子？"，k=3不变）
| chunk_size=4000, overlap=200（默认） | 13 例，带详细解释 |
| chunk_size=1024, overlap=200 | 6 例，一句话简述 |
| chunk_size=4000, overlap=500 | 9 例 |

**结论**
- chunk_size管"每块装多少字"，这里我们设置块变小，导致3块检索回的总信息量从约12000字，降到约3000字，导致答案从13例缩到6例，效果立竿见影
- chunk_overlap管"边界安全"：块与块像瓦片一样压边，每块开头重复上一块的结尾，这是为了防止关键句被拦腰切断。改它 → 块边界移动 → 检索命中的材料变化，就导致答案内容换了一批（9 例），影响更间接
- overlap的取舍：越大句子越安全，但冗余越多、块越多；
- 核心公式：模型看到的信息量 ≈ chunk_size × 检索块数
