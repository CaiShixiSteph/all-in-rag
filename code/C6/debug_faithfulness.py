# debug_faithfulness.py —— 排查忠实度 0% 之谜（用完可删）
# 假设：FaithfulnessEvaluator 把整段回答当一条断言，DeepSeek 法官判罚过严 → 全部 NO
# 方法：只跑 1 个问题，打印法官看到的全部输入和它的原始判决（单变量最小排查）
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.evaluation import FaithfulnessEvaluator, QueryResponseDataset

Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

async def main():
    # 1. 加载文档和数据集（与 01 完全一致）
    reader = SimpleDirectoryReader(input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"])
    documents = reader.load_data()
    dataset = QueryResponseDataset.from_json("./c6_response_eval_dataset.json")
    # queries 是字典（UUID→问题）；第0条是生成器前言泄漏的垃圾题，取第1条真问题
    query = list(dataset.queries.values())[1]

    # 2. 构建句子窗口引擎（与 01 完全一致）
    parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    engine = index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")],
    )

    # 3. 跑 1 个问题，拿到回答和上下文
    response = await engine.aquery(query)
    print("=" * 60)
    print("=== (1) 问题 ===")
    print(query)
    print("\n=== (2) 引擎回答 ===")
    print(str(response))
    print("\n=== (3) 检索到的上下文（窗口替换后）===")
    for i, n in enumerate(response.source_nodes):
        print(f"--- context {i+1} ---")
        print(n.get_content()[:600])

    # 4. 让法官判这 1 条，打印原始判决
    evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    result = await evaluator.aevaluate_response(query=query, response=response)
    print("\n=== (4) 法官原始判决（feedback）===")
    print(result.feedback)
    print(f"\npassing = {result.passing}")

    # 5. 还原法官收到的完整提示词（取证）
    try:
        filled = evaluator._eval_template.format(
            query_str=str(response),
            context_str="\n".join([n.get_content() for n in response.source_nodes]),
        )
        print("\n=== (5) 法官收到的完整提示词（前1200字符）===")
        print(filled[:1200])
    except Exception as e:
        print(f"\n(提示词还原失败: {e})")

if __name__ == "__main__":
    asyncio.run(main())
