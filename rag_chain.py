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
sentence_embed = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)
with open(EMBED_PATH, "rb") as f:
    doc_metas = pickle.load(f)
vector_store = FAISS(index, sentence_embed, doc_metas)
retriever = vector_store.as_retriever(search_type="similarity", k=4)

# 2. LLM – ChatGLM3
from transformers import AutoModel, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

MODEL_PATH = os.getenv("MODEL_PATH") or "THUDM/chatglm3-6b"
DEVICE = 0 if os.getenv("DEVICE", "cpu") == "cuda" else -1

model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).half().cuda() if DEVICE==0 else AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.eval()

llm_pipeline = pipeline("text-generation", model=model, tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True), max_new_tokens=512, do_sample=False, device=DEVICE)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 3. Prompt
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "基于以下已知信息回答用户问题。请用中文回答，若无法从中得到答案，请说明无法从提供上下文中找到答案，不要编造。\n\n"+
        "已知信息：\n{context}\n\n"+
        "用户提问：{query}\n"
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
    return qa_chain.run(query)