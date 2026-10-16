import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 备用方案：hi_res 用本地模型，不走网络下载
os.environ["UNSTRUCTURED_DEFAULT_MODEL_INITIALIZE_PARAMS_JSON_PATH"] = r"C:/Users/蔡仕曦/unstructured_models/model_params.json"
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:/Anaconda/envs/all-in-rag/Library/bin/tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:/Anaconda/envs/all-in-rag/share/tessdata"

from unstructured.partition.pdf import partition_pdf
from collections import Counter

pdf_path = "../../data/C2/pdf/rag.pdf"

elements = partition_pdf(filename=pdf_path, strategy="hi_res",
                         languages=["chi_sim", "eng"])
print(f"hi_res: {len(elements)} 个元素")
print(dict(Counter(e.category for e in elements)))
