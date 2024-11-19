import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from lightrag.utils import EmbeddingFunc

from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
from lightrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer
import os
import chardet
WORKING_DIR = "./2wiki"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


os.environ["OPENAI_BASE_URL"] = "https://api.agicto.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-hisVuPSQjfmRu9evs356q9PAS5fT46D81LNJXGBoz6qYuNzO"
os.environ["HF_TOKEN"]="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"

# 使用检测到的编码读取文件内容
with open('data/2WIKI/output.txt', encoding='utf-8') as f:
     text = f.read()


# Initialize LightRAG with Hugging Face model
from lightrag.llm import ollama_model_complete, ollama_embedding

# Initialize LightRAG with Ollama model
from lightrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer

# Initialize LightRAG with Hugging Face model
from lightrag.llm import ollama_model_complete, ollama_embedding

# Initialize LightRAG with Ollama model
from lightrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer

# Initialize LightRAG with Hugging Face model
# rag = LightRAG(
#     working_dir=WORKING_DIR,
#     llm_model_func=hf_model_complete,  # Use Hugging Face model for text generation
#     llm_model_name='mistralai/Mistral-7B-Instruct-v0.2',  # Model name from Hugging Face
#     # Use Hugging Face embedding function
#     embedding_func=EmbeddingFunc(
#         embedding_dim=384,
#         max_token_size=5000,
#         func=lambda texts: hf_embedding(
#             texts,
#             tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
#             embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#         )
#     ),
# )

from lightrag.llm import ollama_model_complete, ollama_embedding

# Initialize LightRAG with Ollama model
#
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

rag.insert(text)

# Perform naive search
print(rag.query("Who is the father of the director of film O Nanna Nalle?", param=QueryParam(mode="naive")))

# Perform local search
print(rag.query("Who is the father of the director of film O Nanna Nalle?", param=QueryParam(mode="local")))

# Perform global search
print(rag.query("Who is the father of the director of film O Nanna Nalle?", param=QueryParam(mode="global")))

# Perform hybrid search
print(rag.query("Who is the father of the director of film O Nanna Nalle?", param=QueryParam(mode="hybrid")))