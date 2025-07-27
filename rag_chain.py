"""LangChain RAG 检索‑生成链"""
import os, pickle, faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

INDEX_PATH = "data/faiss.index"
EMBED_PATH = "data/embeddings.pkl"

# 1. 向量库封装
# 1. 向量库封装
sentence_embed = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 加载向量和元数据
index = faiss.read_index(INDEX_PATH)
with open(EMBED_PATH, "rb") as f:
    doc_metas = pickle.load(f)

# 构建 FAISS 所需的 index_to_docstore_id 和 docstore
index_to_docstore_id = {i: str(i) for i in range(len(doc_metas))}
from langchain.docstore.in_memory import InMemoryDocstore
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(doc_metas)})

# 正确创建 FAISS 向量数据库
vector_store = FAISS(
    embedding_function=sentence_embed,
    index=index,
    index_to_docstore_id=index_to_docstore_id,
    docstore=docstore
)

retriever = vector_store.as_retriever(search_type="similarity", k=4)


import os
import multiprocessing
from langchain_community.llms import LlamaCpp

model_path = os.getenv("MODEL_PATH", "./models/qwen1_5-4b-chat-q4_k_m.gguf")
n_threads = multiprocessing.cpu_count()  # CPU 线程数

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.01,
    max_tokens=512,
    n_ctx=8192,               # Qwen1.5-4B 支持 32k，可按需调整
    n_batch=512,
    n_threads=n_threads,
    n_gpu_layers=0,          # 有 GPU 时开启；纯 CPU 可设为 0 或删除
    streaming=True,           # 支持流式输出（如用于 Streamlit）
    model_kwargs={"chat_format": "chatml"}  # 关键参数：指定 Qwen 聊天格式
)


# 3. Prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "基于以下已知信息回答用户问题。请用中文回答，若无法从中得到答案，请说明无法从提供上下文中找到答案，不要编造。\n\n"+
        "已知信息：\n{context}\n\n"+
        "用户提问：{question}\n"
    )
)

# 4. QA 链
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    memory=memory,
    chain_type_kwargs={"prompt": prompt_template},
    verbose=True,
)

def chat(query: str):
    return qa_chain.run({"query": query})