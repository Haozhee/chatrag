"""批量加载 data/docs 下的文件，切分后写入 FAISS 向量库"""
import os, glob, faiss, pickle
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DOC_DIR = "data/docs"
INDEX_PATH = "data/faiss.index"
EMBED_PATH = "data/embeddings.pkl"

# 1. 读取 & 切分
loader_map = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}
raw_docs = []
for file in glob.glob(f"{DOC_DIR}/*"):
    ext = os.path.splitext(file)[-1].lower()
    loader_cls = loader_map.get(ext)
    if loader_cls:
        loader = loader_cls(file)
        raw_docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)

# 2. 嵌入
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode([d.page_content for d in docs], show_progress_bar=True)

# 3. 建立 Faiss 索引
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
with open(EMBED_PATH, "wb") as f:
    pickle.dump(docs, f)
print("[Ingest] 索引构建完成，总文档块:", len(docs))