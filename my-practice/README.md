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
