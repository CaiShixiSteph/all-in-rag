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


## 2026-10-15 C1第二个示例：LlamaIndex 版RAG跑通

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

## C1动手练习记录（LangChain版）

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


## 2026-10-16 C2 数据加载：Unstructured解析PDF

### 一、fast策略与Garbage In, Garbage Out
- 01_unstructured_example.py 跑通：partition()把PDF按版面拆成279个元素（7500字符），并给每个元素贴类型标签（Header / Title / NarrativeText / Footer / ListItem / UncategorizedText）
- partition()的作用是：格式解析（二进制PDF → 文字）+ 结构识别（每段贴标签）。
  C1的UnstructuredMarkdownLoader内部就是partition，并且发现rag.pdf是百度百科网页的打印版，包括百度导航栏、PDFCrowd页脚、相关搜索……全被拆出来了 → 垃圾进垃圾出，就是说loader不判断内容好坏，只拆解
fast策略靠字号大小/加粗猜结构，找到了195个假Title，就是说有大量正文被误判，真正的正文只识别出3个，这说明需要优化。

### 二、清洗实验
- 首先我按category过滤，发现行不通，过滤后元素个数从279降到了201，扔掉了22个Header、15个Footer和41个UncategorizedText，但出错的地方在于：
  Element 96「RAG 包含三个主要过程」核心定义被扔
  Element 256「相关搜索：宝宝奶粉质量排行榜」被留（无用的信息被留下）
- 结论：UncategorizedText是个大杂烩筐（按钮数字+广告语+正文混装），说明category标签不可靠，不能一刀切清洗，需要更强的解析策略

### 三、语法
- partition参数：
filename / content_type：MIME 类型，如 application/pdf
file：已打开的文件对象，与 filename 二选一 
url：远程文档，就是可以填入一个网页，让大模型自己去找去下载里面的内容
include_page_breaks：保留分页信息
strategy：可选auto/fast/hi_res/ocr_only
encoding：文本编码，主要对txt有用
sum(len(str(e)) for e in elements)：生成器表达式，先转字符串再求长度再求和
enumerate(elements, 1)：带编号的遍历，每次生成（编号, 元素）这样的元组，
  for i, element in ... 解包接收，编号从1开始

### 四、三种策略对比
fast | ocr_only | hi_res |
文字来源：fast是PDF内嵌文字层（原件），OCR是从图片猜字，类似于誊抄稿，hi_res的是版面模型圈区域+OCR
元素总数对比： | 279 | 87 | 227 |
正文NarrativeText对比：| 3 | 3 | 68 |（hi_res最高，识别出了最多的正文）
Title对比： | 195 | 49 | 30 |（仍是hi_res表现最好，而fast将很多元素误判成了title）
图片/表格/图注对比： | 0 / 0 / 0 | 0 / 0 / 0 | 21 / 4 / 4 |（只有hi_res识别出了表格图片）
文字错字：fast无错字，其余均有
速度对比：fast为秒级，其余为分钟级

### 五、核心结论
1. OCR就是看图片猜字，所以会猜错，举例来说有将网页猜成了BR页，冒充猜成了由充等。所以OCR只配给扫描件或照片型PDF用；电子版PDF有文字层，直接读就行
2. 不可能三角：文字准、结构准、速度快，只能三选二
3. 三样全要的优化方向：版面模型圈区域 + 直接从PDF抽文字层（不做 OCR），查阅得知这就是MinerU、Docling 等专业工具的做法，也是书里说实际项目更多用它们的原因
观察结论：hi_res结构大幅改善（正文68个、识别出图片/表格），但文字仍为OCR结果，是有错字的

### 六、关于安装的问题
- pytesseract未安装，要用 C:/Anaconda/envs/all-in-rag/python.exe -m pip install pytesseract
- 当hf-mirror镜像罢工，备用方案是ModelScope国内镜像下载 yolox_l0.05.onnx到本地
再用环境变量UNSTRUCTURED_DEFAULT_MODEL_INITIALIZE_PARAMS_JSON_PATH指向本地model_params.json（含 model_path + label_map），绕开网络下载直接加载本地模型

## 2026-10-16 C2文本分块：理论总结

### 一、文本分块是什么
把加载后的长文档切成小块。切出来的块是后续向量检索和模型处理的基本单位，也就是说，检索的最小颗粒、喂给LLM的最小上下文，都是块。

### 二、为什么要分块
1. 两个模型的硬限制
- 嵌入模型：有严格的输入长度上限（如 bge-base-zh-v1.5 是512token），超出的部分会被截断，向量无法完整代表原文，块大小必须小于等于嵌入模型窗口
- LLM：上下文窗口有限，检索到的所有块、问题、提示词都要塞进去。单块太大导致能塞的块数变少，导致参考信息广度受限
2.块不是越大越好
- 信息稀释：嵌入 = 分词 → 向量化 → 池化压缩成一个向量。块越长，一个向量要概括的语义点越多，导致表示越笼统，从而导致检索精度下降
LLM处理长上下文时倾向记住开头和结尾而忽略中间，这样一来关键信息就会淹没在噪音里
- 主题稀释：一个块塞多个主题导致相关性被稀释，检索召不回。
  例子：鲁班七号攻略把"技能+出装+背景故事"塞一块，
  玩家问"怎么出装"时这个大杂烩块得分低，召不回来；
  切成三个主题块后，"出装块"精准命中

一句话总结：分块的目标是每块大小适配模型上限，以及每块主题聚焦，和关键句不被切断。

## 2026-10-16 C2文本分块

### 一、三种策略对比
| 02 字符分块 | 03 递归分块 | 04 语义分块(percentile) |
块数： | 14 | 22 | 2 |
判断依据：02只认段落（\n\n），03只认长度+分隔符层级，04认句间语义差异
暴露的问题：字符分块不切长段落，有超长块警告，但是不会分割；递归分块切出12字符碎片块，语义分块断点少、位置怪（切在目录中间）
速度对比：字符分块瞬间完成（字符串操作），递归分块也是瞬间，语义分块最慢（每句过一次嵌入模型）

### 二、SemanticChunker原理
- 计算对象：相邻句子的嵌入向量之间的余弦距离（差异值）
- buffer_size就是取景框半径：嵌入每句时带上前后各buffer_size句一起嵌入。
  buffer_size=1是前后各1句，2是前后各2句
- 更长的组合文本有什么用：让句子的向量"带着上下文背景"，比较的是带背景的语义，而不是孤句的字面意思
- "既然算的是相邻句子差异，buffer_size 有什么用？"——
  buffer_size改变的是每句的嵌入值本身，嵌入值变了→差异值变→断点变。断点判定是在这些差异值上再筛一遍
- 换断点方法：只改breakpoint_threshold_type的值（四种：percentile / standard_deviation / interquartile / gradient）

buffer_size 决定什么？
- "相邻句子"永远指位置上紧挨的两句（句i和句i+1），不随buffer_size变；断点永远在位置相邻的两句之间判断
- buffer_size 决定的是每句算向量时的取景框半径：
  buffer_size=1 → 向量 =【上一句 + 本句 + 下一句】拼起来嵌入；
  buffer_size=2 → 向量 =【上两句 + 上一句 + 本句 + 下一句 + 下两句】
- 所以buffer_size管的是"每句向量带多少上下文"，不管"谁跟谁比"

拼出来的临时长文本有什么用？
假设话题A的句子A1 A2 A3，话题B的句子B1 B2，真断点在A3和B1之间：
- 没有buffer（每句单独嵌入）：A3向量 = embed(A3)，B1向量 = embed(B1)，差异就是字面差异。
但问题是A3可能是A话题的总结句、B1是B话题的引入句，两者字面可以很像，比如他们都以"总的来说""另一方面"开头，真断点处差异值反而小；反过来话题 A内部相邻两句可能一句讲技能、一句讲数值，字面差很大，可能导致假断点。这个时候单句嵌入噪声大，这就是C2里short text向量质量差的毛病
- 如果有buffer_size=1：A3向量 = embed(【A2 + A3 + B1】)，B1向量 = embed(【A3 + B1 + B2】)，每个向量不再代表孤立句子，而代表"这个位置的话题面貌"。话题A内部相邻两句的取景框高度重叠（都是A的内容），差异小；跨话题两句的取景框里混进不同话题，差异大。于是差异曲线平滑可信：真断点突出，假波动被压平。最后算的是每对相邻句子的嵌入向量之间的余弦距离
- 类比：找一年的"季节转换点"。用每日温度画曲线就全是毛刺，看不出哪天换季；如果用7日滑动平均画曲线就光滑了，入秋点一目了然。buffer_size = 滑动窗口半径，不是改"哪天跟哪天比"，而是把每天的温度先改成可信数值再比

### 三、四种断点方法实验
percentile：切出2块，取差异最大的前5%，按比例硬切，永远有断点
standard_deviation：切出1块，均值+3σ门槛，没有异常差异就不切
interquartile：切出1块，四分位距找异常值，同上，更稳健
gradient：切出2块，找差异突增的坡度，最敏感
观察：两个切了的方法断点都落在目录区（一次切条目之间，一次切进条目内部），因为单主题文档差异整体又小又平稳，断点不可解释。

## 2026-10-16 C2分块理论补充：3.4 结构分块 + 四 开源框架分块

### 一、3.4基于文档结构的分块（MarkdownHeaderTextSplitter）
- 原理：先按标题分组，再按需细分。先定义标题映射（# → 一级标题，## → 二级标题），然后把每个标题下的内容聚合在一起，直到遇到下一个同级或更高级标题
- 核心优势：元数据注入，每块自带"地址"，如 {"Header 1": "第三章：模型评估", "Header 2": "3.2节：评估指标"}，这样就把标题先提取出来了，后面再对正文做处理。给LLM提供了信息片段的来源背景
- 局限与组合：某个章节可能超长，此时就需要两阶段组合：
  第一步MarkdownHeaderTextSplitter按标题切大逻辑块，然后再第二步RecursiveCharacterTextSplitter切小块，小块继承第一步的标题元数据（处理结构化文档做RAG的理想方案）

### 二、其他开源框架中的分块策略
#### 4.1 Unstructured
- 流程：先分区（partition产出带标签的元素）→ 再分块，"先理解、后分割"
- basic（默认）：连续合并元素直到达到max_characters上限，尽量填满每一个块
- by_title：把Title元素当新章节起点，强制开新块，块内不跨章节

#### 4.2 LlamaIndex：一切皆节点
- 分块看作节点转换（Transformation）的一环，分为三类解析器：
  - 结构感知型：MarkdownNodeParser / JSONNodeParser / CodeSplitter（懂标题、懂代码函数）
  - 语义感知型：SemanticSplitterNodeParser（≈LangChain 的 SemanticChunker）；
    SentenceWindowNodeParser：每句一个节点，元数据存前后N句窗口 → 检索时先用单句嵌入精确匹配，
再送带窗口的全文给 LLM（和buffer_size同源，但窗口放在检索阶段用）
  - 常规型：TokenTextSplitter / SentenceSplitter
- 流水线：MarkdownNodeParser按章节切 → SentenceSplitter句子级再切，元数据贯穿始终
- 互操作：LangchainNodeParser可以把LangChain的分块器包进LlamaIndex用

#### 4.3 ChunkViz
- 网页版分块可视化工具：输入文档+分块配置，用彩色块展示每块的边界和重叠，用来直观理解分块逻辑


## 2026-10-19 C3

### 第一节 向量嵌入

**一、基础概念**
- Embedding：把真实世界的高维数据（文本/图像/音频）变成低维、稠密的数值向量。
  每个词/段落/图片在多维空间里有一个"坐标"，这个坐标就是向量，它嵌入了原始数据的关键信息
- 三要素：数据对象 + Embedding模型 + 输出向量（固定长度一维数组，通常几百到几千维）
- 语义空间核心原则：语义相似 → 向量距离近；语义不相关 → 距离远
- 三种度量：余弦相似度（最常用，越接近1越相似）、点积（向量归一化后等价于余弦）、欧氏距离（越小越相似）。C2语义分块找断点算的"余弦距离"就是这里的余弦相似度

**二、Embedding在RAG中的作用**
- 离线建索引：文档切块 → 每块转向量 → 存向量库
- 在线检索：用户问题用同一个模型转向量 → 与所有块向量算相似度 → 召回Top-K送LLM
- 关键：问题和文档必须用同一个模型嵌入（C1笔记"同一个翻译员才能比"就是这个原理）
- 质量决定一切：差模型召回错内容 → 污染上下文 → 答案差（垃圾进垃圾出在检索环节的体现）

**三、训练原理**
- MLM 掩码语言模型：随机盖住15%的词让模型猜，模型被迫学习词与上下文的关系
- NSP下一句预测：判断B是不是A的下一句，后来发现任务太简单甚至有害，RoBERTa/SBERT移除了它
- 度量学习：直接以"相似度"为优化目标，拉近正例对、推远负例对
- 对比学习：三元组 (Anchor, Positive, Negative)，拉近锚点与正例、推远锚点与负例

**四、选型指南**
- 起点：MTEB排行榜（Hugging Face维护的文本嵌入评测基准）
- 四维图读法：横轴=参数量（越大越强越贵）、纵轴=平均任务得分、
  气泡大小=嵌入维度、气泡颜色=最大处理长度
- 重点维度：Task（RAG看Retrieval排名）、Language（中文支持）、Size、Dimensions、Max Tokens、Cost
榜单不是最终答案，对于自己制作的模型，重点是选基线、建私有评测集，然后迭代测试优化

### 第二节 多模态嵌入

**一、为什么需要**
- 文本向量和图像向量在隔离空间，中间有一堵"模态墙"：文本嵌入无法理解"那张有红色汽车的图片"这种查询
- 目标：把图文音视频映射到同一个共享向量空间（"一只奔跑的狗"的文字向量 ≈ 狗奔跑图片的向量）
- 关键挑战：跨模态对齐

**二、CLIP（OpenAI，图文多模态范式开创者）**
- 双编码器架构：图像编码器 + 文本编码器，各自映射到同一个共享向量空间
- 训练：对比学习——最大化正确图文对的相似度，最小化所有错误配对的相似度
- 零样本能力：分类任务变图文检索（判断图片是不是猫 = 算图片向量与"a photo of a cat"的相似度），无需微调

**三、bge-visualized-m3（BAAI，BGE-M3 文本底座 + 图像能力）**
- M3 = 多语言（100+语言）/ 多功能（密集检索 + 多向量检索）/ 多粒度（短句到 8192 token）
- 架构：视觉编码器提取图像patch token → 映射成图像token → 与文本token一起进BGE的Transformer联合建模

**四、代码示例 01_bge_visualized.py**
- encode 三种模式：纯文本 / 纯图像 / 图文联合
- sim = a @ b.T：矩阵乘算余弦相似度（向量已归一化，点积=余弦）

针对代码，我做了三次实验，替换文字：logo文本 / blue whale / apple
| 对比 | logo文本 | blue whale | apple |
| 纯图 vs 纯图 | 0.8318 | 0.8318 | 0.8318 |
| 图文1 vs 纯图1 | 0.8291 | 0.9218 | 0.8279 |
| 图文1 vs 纯文本 | 0.7627 | 0.7572 | 0.7068 |
| 图文1 vs 图文2 | 0.9058 | 0.8719 | 0.8926 |

结论：
1.图文结合向量里图像成分占大头，所以即使apple和图片完全不搭，图文vs纯图仍有0.83
2.同样的错误文字也反而拉高了相似度，因为两个向量都装着同一个apple，都朝apple方向偏同样一截，所以图文1vs图文2反而被拉高，这就好比两个人都戴错同一顶帽子，站一起反而更像了
3.文字贴合图（blue whale）的图文vs纯图相似度达到0.92；文字不搭的apple回落到0.83；

## 2026-10-19 C3 第三节 向量数据库：FAISS

### 一、内容总结
- 要解决的问题：向量从几百条涨到几亿条，如何快速找到与查询最相似的那几个？
  方法是用专门的索引技术（HNSW、IVF 等）做近似最近邻（ANN）搜索，毫秒级返回
- 主要功能：高效相似性搜索、高维向量存储管理（增删改查）、丰富查询能力（如标量过滤 year>2023）、可扩展高可用、与AI框架生态集成
- 与传统数据库（MySQL）的区别：
  向量数据库的核心数据类型是高维向量，传统数据库是结构化数据（文本/数字/日期）；查询方式上，前者是相似性搜索（"最像的3个"），后者是精确匹配（WHERE age=25；向量数据库采用的索引机制是HNSW / IVF / LSH，而传统数据库多利用B-Tree / Hash。二者是互补关系：业务元数据存MySQL，AI产生的海量向量存向量库
- 工作原理：四层架构（存储层→索引层→查询层→服务层）；
  四类索引技术：树（Annoy随机投影树）、哈希（LSH相似向量进同一桶）、图（HNSW多层邻近图）、量化（IVF/PQ聚类压缩）
- FAISS的特殊身份：本质是算法库，不是数据库服务，因为索引直接存成本地文件（.faiss + .pkl），没有服务器，轻量高效

### 二、02_langchain_faiss.py代码解读
- 准备数据：texts是3条示例文本模拟知识库；
  docs = [Document(page_content=t) for t in texts] 是列表推导式，把每条纯文本包装成Document对象
- 为什么Document(page_content=t)就可以遍历texts？
  循环由for t in texts负责，Document(page_content=t)只是每轮执行的动作（把t包装成 Document）。page_content=t：Document(...)是构造调用，page_content是它的参数名（"正文内容"），把t传进去，也就是page_content=正文内容
- embeddings：翻译员，把文本翻译成512维向量
- 核心行vectorstore = FAISS.from_documents(docs, embeddings)：一步建库，内部四层嵌套
  ① from_documents（封装层：提取纯文本和元数据）
  ② from_texts（向量化入口：embed_documents 批量翻译成向量）
  ③ __from（建空框架：默认 L2 欧氏距离初始化空索引，准备 docstore 和 id 映射）
  ④ __add（填充：向量转 numpy 数组进索引、原文存 docstore、建立"整数编号→文档"映射）
- vectorstore.save_local("./faiss_index_store")：持久化——生成两个文件：
 index.faiss（向量索引）+ index.pkl（文档原文+id映射）；./ 表示当前目录
-重载FAISS.load_local(path, embeddings,
					   allow_dangerous_deserialization=True)：
  必须传同一个embeddings（换了翻译员坐标体系不同没法比）；
  allow_dangerous_deserialization=True是显式确认pkl文件可信（pkl反序列化可以执行任意代码）
- similarity_search(query, k=1)：问题向量化 → ANN 搜索 → 返回最像的k个Document

### 三、练习记录
**练习1：跑通03，查看LlamaIndex的JSON存储**
- 生成llamaindex_index_store/ 五个JSON文件：
  - default__vector_store.json（34KB）：向量仓库——embedding_dict里 键=每篇的UUID编号，值=512维向量；
    text_id_to_ref_doc_id是"节点编号→原文档编号"映射；metadata_dict存元数据
  - docstore.json：文档仓库——原文 + doc_hash（内容指纹）
  - index_store.json：索引登记表——加载时靠它知道该读哪些文件
  - graph_store / image__vector_store：空
- 对比FAISS：二进制2个文件 vs JSON 5个文件（可读）；整数编号0,1,2 vs 36位UUID；
  本质是同一设计，都是向量+原文成对存、靠编号相连

**练习2：自己写加载+检索脚本**
- 加载三步：
Settings.embed_model请同款翻译员
StorageContext.from_defaults(persist_dir=...)当"地图"
load_index_from_storage按图重建索引
- 存读对称：作者的代码用VectorStoreIndex.from_documents建，我这里用load_index_from_storage重建；
persist写，通过StorageContext.from_defaults找回来
- as_retriever(similarity_top_k=1)：只检索不生成的纯入口，不需要LLM/API key（对比C1的as_query_engine是检索+生成打包）
另外还发现离线开关不是万能的
- llama_index的嵌入类不吃离线开关（加了HF_HUB_OFFLINE=1仍然联网检查报错），但吃镜像地址
- 解决方法：改用os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

## 2026-10-21 C3第四节：Milvus与多模态检索实践

### 一、Milvus是什么
- 开源分布式向量数据库
- 与FAISS的本质区别：FAISS是库，要import进代码里用，Milvus是服务，可以独立运行、7×24在线，Python代码是客户通过localhost:19530访问它
- 能处理十亿级向量，高可用、易扩展

### 二、Docker部署
 milvus-standalone，职责是向量检索主程序，类比于业务部门
 milvus-etcd，职责是存元数据（有哪些 Collection、Schema），类比于档案室 
 milvus-minio，职责是对象存储（真正的数据文件），类比于仓库

- docker-compose.yml = "开厂清单"（三个部门各自的镜像、端口、数据盘）
- `docker compose up -d` = 一键开厂；`down` = 关门留数据；`down -v` = 拆厂清数据

### 三、核心组件
- **Collection**（集合），可类比成图书馆，约等于关系型数据库的"表"，一切数据操作围绕它展开
- **Partition**（分区），就类似图书馆里的区域（小说区/科技区），查询时只搜指定分区可大幅提速，最多1024个
- **Schema**（模式），类比图书卡片规则，规定每条数据登记哪些字段，Collection也是由Schema定义的，这就是为什么档案室里必须有Schema。
Schema规定了Collection的数据结构，定义了其中包含的所有字段 (Field)及其属性。一个设计良好的Schema能够保证数据一致性并提升查询性能，通常包含以下字段：
  - 主键字段（必须、唯一）｜向量字段（存储核心数据，一个Collection可以有一个或多个）｜标量字段（元数据，用于过滤查询）
- **Entity**（实体），类似一本具体的书（一条数据）
- **Alias**（别名），类似动态推荐书单，指向某个Collection，方便应用层调用，实现数据更新时的无缝切换，更新数据时先建好collection_v2，再把别名原子性切换过去，无需改代码，好处是当我们需要对一个在线服务的Collection进行大规模的数据更新或重建索引。直接在原Collection上操作风险很高。正确的做法是：
1. 创建一个新的Collection (collection_v2) 并导入、索引好所有新数据。
2. 将指向旧Collection (collection_v1)的别名（例如 my_app_collection）原子性地切换到新Collection (collection_v2)上。

### 四、索引类型与选择
| 索引 | 原理 | 优点 | 缺点 | 适用 |
| FLAT | 暴力搜索全部向量 | 100%召回 | 慢、吃内存 | 小数据、要精度 |
| IVF系列 | 聚类分"桶"，只搜最近的桶 | 快，平衡之选 | 召回非100% | 通用大规模 |
| HNSW | 多层邻近图 | 极快、召回高 | 内存占用大、建索引慢 | 低延迟实时场景 |
| DiskANN | 建在SSD上的图索引 | 突破内存容量 | 延迟稍高 | 十亿级数据 |
选择索引要在数据规模、内存限制、查询性能、召回率之间权衡，没有唯一答案，实测定参数。

### 五、检索
- **基础ANN检索**：近似最近邻，利用预先构建好的索引，能够极速地从海量数据中找到与查询向量最相似的Top-K个结果。
核心参数：`anns_field`（在哪个向量字段搜，当涉及到多模态有图片有文字的时候就需要这个参数告诉模型去图还是去文本搜）、`data`（查询向量）、`limit/top_k`（返回几个）、`search_params`（指定查询的metric_type，可以用COSINE、L2、内积；还有params，这个根据不同的索引有不同的参数）
- **过滤检索**：向量相似性 + 标量过滤（相似的红裙子，且价格小于500），就是带过滤条件的ANN
- **范围检索**：返回相似度落在阈值区间的所有结果（人脸识别：相似度>0.9），这是因为有时我们关心的不仅仅是最相似的Top-K个结果，而是符合范围的所有
- **混合检索**：多个向量字段并行检索 + Reranker 融合结果（RRFRanker 平衡 / WeightedRanker 加权）
1. 并行检索：应用针对不同的向量字段（如一个用于文本语义的密集向量，一个用于关键词匹配的稀疏向量，一个用于图像内容的多模态向量）分别发起ANN检索请求。
2. 结果融合 (Rerank)：Milvus 使用一个重排策略（Reranker）将来自不同检索流的结果合并成一个统一的、更高质量的排序列表。常用的策略有 RRFRanker（平衡各方结果）和 WeightedRanker（可为特定字段结果加权）。
例如：
"安静舒适的白色耳机"同时查文本向量和图片向量
- **分组检索**：按字段分组去重，解决"前10名全是同一本书的不同章节"，也就是检索结果多样性不足问题，我们想确保返回的结果来自不同的书籍或来源，就用分组检索。原理是分组检索允许指定一个字段（如document_id）对结果进行分组。Milvus会在检索后，确保返回的结果中每个组（每个document_id）只出现一次（或指定的次数），且返回的是该组内与查询最相似的那个实体。

补充：密集向量vs稀疏向量
密集向量（Dense）：把一整句话或一张图压缩成若干个小数（比如BGE的768维）。每个数字单独看没有任何含义，是模型吃了海量语料学出来的压缩码。所以它能捕捉语义。它就像印象派画家，远看神似，能认出"画的是个人"，但细节糊了
稀疏向量（Sparse）：维度 = 整个词表大小（几万到几十万维），但绝大部分是0，只有文本里真正出现的那些词对应的维度有数值（通常是词频加权）。它本质是高级版关键词匹配。经典代表BM25，神经网络版有SPLADE。可以逐字对照，一个标点都不放过，但不懂言外之意
密集向量的强项（懂意思、联想、同义）：
搜"手机没电了怎么办"，能匹配到讲"充电宝/续航"的文档（意思相近但字完全不同）
跨语言："apple"和"苹果"能对上
多模态：图和文在同一个空间里比
密集向量的弱点：
搜"iPhone15"可能把"iPhone14"也排前面（语义都差不多）
搜"SKU-8848"、"ERR_502"、法律条文编号，这些语义帮不上忙，只能抠字面的场合，密集向量表现很差
稀疏向量正好补上（懂字面、精确、罕见词）：
精确词、编号、代码、罕见术语都可一字不差地命中
可解释：能告诉你"是因为都出现了'违约金'这个词才匹配的"
弱点：不懂同义词，搜"汽车"匹配不到只写了"车"的文档

### 六、多模态实践04
流程：
1. 初始化：加载Visualized-BGE模型（Encoder类）和MilvusClient（连 localhost:19530）
2. 建Collection：Schema = 自增id（主键）+ vector（768维向量）+ image_path（标量，最长512字符）
3. 插入数据：7张图逐一 `encode_image`成向量，连同路径批量insert
4. 建索引：HNSW + COSINE（M=16：每节点最多16个邻居；efConstruction=256：建索引时搜索范围）
5. `load_collection`：加载进内存才能检索，否则只能在磁盘上读不到
6. 检索：查询图 + 文字"一条龙" → `encode_query` 图文联合向量 → search Top5
7. 可视化：查询图（蓝框）+ 检索结果拼成一张对比图存search_result.png
8. 清理：release（搬下内存）+ drop（删除 Collection），保证下次跑是干净状态
部分代码解释：
- **Encoder 两个方法**：`encode_query`（图文联合：图片+文字一起编码成一个向量）vs `encode_image`（纯图）。图库存的是纯图向量，查询用的是图文联合向量（图+"一条龙"），两种配方不同，这在结果上可以看到，Top1是自己也只有0.9466而不是1.0，因为图文混合带上了文字，和纯图不一样
- `torch.no_grad()`：不算梯度，省内存省时间（只有训练才需要梯度）；`model.eval()`：切到"考试模式"，关掉Dropout等只在训练时生效的东西
- **visualize_results细节**：`np.full` 铺一张全白画布再贴图；`[:, :, ::-1]` 是RGB→BGR反转（PIL读图是RGB顺序，OpenCV存图是BGR，不转颜色会变）；`copyMakeBorder`加边框（查询图蓝框、结果图黑框）；`i//3` 和 `i%3` 把第i张图定位到3列网格；`np.hstack`左右拼接成一张大图
- `dim = len(encoder.encode_image(image_list[0]))`：先拿第一张图跑一次量出维度（768），不用手填，换模型也不会错
- `search(...)[0]`：结果按查询向量分组，传几个查询向量就有几组，这里只传1个所以取第 0 组
- 最后release（搬下内存）+ drop（删除）

总体流程：
1. 拿起对讲机：
milvus_client = MilvusClient(uri="http://localhost:19530")
2. 建图书馆（先拆旧的，再建新的）
如果 COLLECTION_NAME 已存在，先 drop 掉（has_collection, drop_collection）
然后到fields（Schema结构）
3. 用Schema创建Collection
create_collection，然后再批量插入
data_to_insert、milvus_client.insert
4. 建检索卡 + 搬上阅览桌
prepare_index_params 后 add_index
然后create_index、load_collection
5. 检索
search_results = milvus_client.search
6. 清场
release_collection、drop_collection

## 2026-10-22 C3第五节

### 一、第五节 1.2 理论：句子窗口检索
- 解决的矛盾（C2 分块两难的解法）：块小→检索精确但LLM没上下文；块大→上下文足但向量被稀释
- 核心思想：索引小块、搜索小块、交付大块——搜索阶段用小卡片保证命中精准，交付阶段把命中的卡片换成大卡片保证上下文丰富。
- 四步流程：索引（每句一个节点，窗口存metadata）→ 检索（只搜单句向量）→ 后处理（MetadataReplacementPostProcessor 把text替换成窗口）→ 生成
- 节点结构：`text`=单句（嵌入用）；`metadata['window']`=前后各N句（生成用）
- 关键：excluded_embed_metadata_keys / excluded_llm_metadata_keys（黑名单机制）：
  - 嵌入模型和LLM都"看不见" window，不排除的话，相邻句子的窗口高度重叠（7句共用6句）→ 向量雷同 → 检索失效；LLM 还会收到重复内容
  - window是节点身上的"暗格"，只有后处理器有钥匙，只在生成前打开一次
- 类比：图书卡正面印一句话摘要（检索用），背面完整上下文贴封条"非后处理器不得翻阅"

### 二、1.2代码要点
**先立核心思想——大小块分工：**
一句话：**索引小块、搜索小块、交付大块。**
同一个节点小块和大块同时存在：`text` = 单句（小块，给搜索用）；`metadata['window']` = 前后各3句拼成的段落（大块，给LLM用）。谁看谁、什么时候看，由两条黑名单 + 后处理器精确控制。

**建句子窗口索引（小块）**
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,                              # 窗口半径：前后各3句
    window_metadata_key="window",               # 大块存进 metadata 的哪个键
    original_text_metadata_key="original_text", # 原句备份到哪个键
)
sentence_nodes = node_parser.get_nodes_from_documents(documents)
sentence_index = VectorStoreIndex(sentence_nodes)

作者补充了 `SentenceWindowNodeParser` 的**源码四步**——这就是"大小块同体"的生产过程：
1. **句子切分**：`sentence_splitter(doc.text)` 把整篇文档切成句子列表（小块原料）
2. **创建基础节点**：`build_nodes_from_splits` 为每个句子建一个TextNode，此时每个节点的`text`就是那个句子（小块）
3. **构建窗口并填充元数据（主循环）**：遍历所有节点，对第i个节点执行：
   - **定位窗口**：列表切片`nodes[max(0, i-window_size): min(i+window_size+1, len(nodes))]`——取"前3句+自己+后3句"；`max(0,...)` 和 `min(...,len)` 两个夹子专门处理开头结尾的越界问题（大块的裁剪）
   - **组合窗口文本**：把窗口内所有句子的text用空格拼成一个长字符串（大块）
   - **填充元数据**：长字符串存进`metadata['window']`，原句存进 `metadata['original_text']`
4. **设置元数据排除项**
`node.excluded_embed_metadata_keys.extend([...])`和`node.excluded_llm_metadata_keys.extend([...])`两条黑名单，确保嵌入模型和LLM都"看不见" window/original_text，只有后处理器有钥匙

**常规分块索引（对照组）**
base_parser = SentenceSplitter(chunk_size=512)
base_nodes = base_parser.get_nodes_from_documents(documents)
base_index = VectorStoreIndex(base_nodes)
512字符一块， 没有小块大块的分工，这就是对照组要暴露的问题。实验设计：同文档、同问题、同top_k，只差索引策略一个变量。

**查询引擎**
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

- `similarity_top_k=2`：召回2个最相关的小块（单句节点）
- `node_postprocessors`：检索后处理器：
`MetadataReplacementPostProcessor`读取节点的`metadata['window']`，用它替换节点的`text`。从此节点在LLM眼里就是大块，这里就是出现差异的核心
- base引擎没有后处理器：LLM直接收到2个512字符块

| 阶段 | 用小块还是大块 | 谁在做 |
| 建库嵌入 | 小块（单句向量） | 嵌入模型（黑名单保证纯净） |
| 相似度搜索 | 小块（单句向量） | 检索器 |
| 后处理 | 小块→大块的交接| MetadataReplacementPostProcessor |
| 生成答案 | 大块（窗口段落） | LLM |
搜的时候用"目录卡片"（小块），拿走的时候是"整本书"（大块）；中间那个"换书"动作就是后处理器。

### 三、实验输出对比（窗口版 vs 常规版）
- 共同点：核心事实4项全命中——检索阶段命中的是同一批核心句子（检索精度的功劳）
- 差异在主题句：窗口版 "uncertainty and limited observational records" 更贴原文（站在段落高度组织答案）；常规版 "detecting and attributing changes" 是LLM自己概括转述


## 2026-10-23 C3第五节（2）：结构化索引——递归检索

### 一、本节笔记：递归检索解决什么问题
问题：异质数据（多张Excel表、多个文档、数据库……）混在一起时，全部打碎塞进一个索引，检索精度会很差，向量相似度分不清"哪个问题该去哪个数据源找答案"。解决思路是采用总分结构 + 路由。
具体来说：
1. 每个子数据源（每张表）生成一个摘要summary
2. 摘要做成指针节点（IndexNode），挂上`index_id`（分机号）
3. 顶层只建一个向量索引，里面只存这些摘要（轻量、专一）
4. 查询时：顶层检索定方向 → 顺着指针到子层 → 子层用自己的引擎真正回答

可类比为四个角色：
| 角色 | 类 | 职责 |
| 总机 | `VectorStoreIndex`（顶层） | 把问题向量化，和摘要比相似度，决定该找谁 |
| 接线员 | `RecursiveRetriever` | 识别命中的是不是指针节点，是就顺着index_id转接 |
| 业务员 | `PandasQueryEngine` | 把自然语言翻译成pandas代码并执行，真正给答案 |
| 路标 | `IndexNode` | text 是摘要，index_id指向子引擎（自己不回答问题） |

### 二、06 代码详解
**循环四张表（21-40 行）**：
xls = pd.ExcelFile(excel_file)          # 打开整个Excel文件
for sheet_name in xls.sheet_names:   # 遍历：年份_1994~年份_1997四张表
    df = pd.read_excel(xls, sheet_name=sheet_name)
    query_engine = PandasQueryEngine(df=df, llm=Settings.llm, verbose=True)  # 业务员
    summary = f"这个表格包含了年份为{year}的电影信息..."   # 摘要
    node=IndexNode(text=summary, index_id=sheet_name)      # 路标
    df_query_engines[sheet_name] = query_engine  # 电话簿：分机号 → 业务员

关键：index_id=sheet_name（即"年份_1994"）和电话簿的key必须一致，接线员就是靠这个id去查电话簿找业务员的。
顶层索引（43行）：VectorStoreIndex(all_nodes) —— 总机，只索引4个摘要节点。
递归检索器（47-55行）：
vector_retriever = vector_index.as_retriever(similarity_top_k=1)  # 顶层找最像的1个
recursive_retriever = RecursiveRetriever(
    "vector",                                   # 根节点id：从哪开始
    retriever_dict={"vector": vector_retriever},# 顶层检索器
    query_engine_dict=df_query_engines,         # 电话簿
    verbose=True,                               # 打印转接记录
)
包装执行（58-64 行）：RetrieverQueryEngine.from_args(recursive_retriever) 把接线员包装成标准查询引擎，query提问、response回答。

可以用打电话来记
打10086："1994年评分人数最少的电影是哪一部？"（第1行，总机VectorStoreIndex接线） 
话务员："转接中，进入1994专席"（第2行RecursiveRetriever，entering） 
专席客服拿起电话，把刚才的话原封不动复述一遍确认："1994年评分人数最少的电影是哪一部？"（第3行，业务员PandasQueryEngine接单） 
然后他查表、报出答案："燃情岁月。"

### 三、2.2 笔记：路由与检索彻底分离
动机：2.1的PandasQueryEngine靠`eval()`执行LLM现场生成的代码，能执行任意Python，生产环境不敢用。2.2换一条更安全的路，完全不写代码，把"该找谁"和"怎么答"拆成两个独立的索引。

**两个索引 + 两步查询**：
| | 索引 | 装什么 | 职责 |
| 路由器 | 摘要索引 | 每张表一句摘要Document| 只回答：该找哪张表 |
| 问答库 | 内容索引 | 整张表转成文本，挂 `metadata={"sheet_name": "年份_1994"}` | 真正提供答案 |

1. 路由：问题在摘要索引检索，拿到元数据里的sheet_name
2. 检索：在内容索引检索，加元数据过滤器（ExactMatchFilter），强制只在 `sheet_name == "年份_1994"` 的文档里搜

**2.1 vs 2.2对比**：
| | 2.1 递归检索 | 2.2 分离式两步检索 |
| 路由方式 | 指针节点（IndexNode.index_id），框架自动转接 | 独立摘要索引 + 自己写代码读metadata |
| 回答问题 | LLM写代码精确计算（idxmin） | 整表文本交给LLM阅读理解|
| 安全性 |  eval()有隐患 | 无代码执行 |
| 适合 | 表大、需要精确计算 | 表小、整表能被LLM读完 |

### 四、07代码详解
**数据清洗段（29-31行，2.1没有的新增）**：
df['评分人数'] = df['评分人数'].astype(str).str.replace('人评价', '').str.strip()
df['评分人数'] = pd.to_numeric(df['评分人数'], errors='coerce').fillna(0).astype(int)

原数据是"102345人评价"字符串。2.1不需要清洗（LLM写代码时自己处理）；2.2 把表转成纯文本让LLM读，清洗后文本更干净。errors='coerce' = 转不动就填NaN的容错写法。
双文档列表：2.1的路标是IndexNode（带index_id指针，框架自动转接）；2.2的路标是普通Document（文本 + 标签），转接逻辑由我们自己写，这就是"分离"的含义。df.to_string(index=False) 把整张DataFrame拍平成一个字符串。

输出对比：07的回答比06更丰富，06只答"燃情岁月"（代码只返回了片名列的值），07答"《燃情岁月》，评分人数为176801"。因为整张表文本都交给了LLM，它阅读时顺手把数字也念出来了。这就是"精确计算"与"阅读理解"两种不同方式的直观体现。

### 五、为什么2.1要执行代码，2.2不用
2.1：问题是计算题，算的是计算机里的pandas程序，答案藏在计算结果里，代码必须被eval()执行。危险在于执行的是LLM现场生成、受用户输入左右的代码（因为有提示词注入，还可能有一些恶意问题的输入导致生成恶意代码，这样一来就会导致回答的结果错误甚至系统被破坏）
2.2：建索引时数据就已转成文本（我们自己的代码，查询前跑一次），查询阶段LLM只读不算，输出从头到尾只是文本，最坏结果也只是答案念错了
账本比喻：2.1是让会写代码的员工写程序算（快准，但可能被教唆夹带私货）；2.2是把账本摊桌上用眼睛找（安全，但账本大了读不动）
关键：危险的不是"执行代码"本身，而是执行LLM查询时临场生成的代码

