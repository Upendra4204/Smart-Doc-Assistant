from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import shutil
import tempfile
from pathlib import Path
import os
import pandas as pd

# Groq + LangChain
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment
load_dotenv()
api_key = os.getenv("API-KEY")

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

# Groq client + embedding model
client = Groq(api_key=api_key)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS vectorstore (or initialize empty)
FAISS_INDEX_PATH = "vector_db/faiss_index"
try:
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
except:
    vectorstore = None

@app.get("/", response_class=HTMLResponse)
def serve_home(request: Request):
    return templates.TemplateResponse("index1.html", {"request": request})


class QueryRequest(BaseModel):
    question: str

def get_context_from_faiss(query: str, k: int = 4):
    if not vectorstore:
        return "⚠️ No documents indexed yet."
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

@app.post("/ask")
async def ask_question(query: QueryRequest):
    user_question = query.question
    context = get_context_from_faiss(user_question)

    system_prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {user_question}"""

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
    )

    reply = completion.choices[0].message.content
    return {"question": user_question, "answer": reply}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    # Load docs
    docs = []
    try:
        if suffix == ".txt":
            docs = TextLoader(str(tmp_path)).load()
        elif suffix == ".pdf":
            docs = PyPDFLoader(str(tmp_path)).load()
        elif suffix == ".csv":
            docs = CSVLoader(str(tmp_path)).load()
        elif suffix == ".docx":
            docs = Docx2txtLoader(str(tmp_path)).load()
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(tmp_path)
            for _, row in df.iterrows():
                row_text = " | ".join(str(cell) for cell in row if pd.notna(cell))
                docs.append(Document(page_content=row_text, metadata={"source": str(file.filename)}))
        else:
            return {"status": "❌ Unsupported file type choose pdf/txt/csv/docx/xlsx/xls file extensions"}
    except Exception as e:
        return {"status": f"❌ Failed to load file: {e}"}

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embed + save
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(FAISS_INDEX_PATH)

    return {"status": "✅ File processed and indexed successfully!"}
