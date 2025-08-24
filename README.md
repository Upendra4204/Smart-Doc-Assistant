# 🧠 RAG Chatbot with FastAPI + Groq + LangChain

A Retrieval-Augmented Generation (RAG) chatbot built using **FastAPI**, **LangChain**, **FAISS**, and **Groq LLMs**.  
This chatbot allows you to **upload documents (PDF, DOCX, TXT, CSV, XLSX, XLS)**, process them into embeddings, and then ask **natural language questions** to retrieve answers from your documents.

---

## ⚡ Features
- 📂 Upload multiple document types: `.pdf`, `.docx`, `.txt`, `.csv`, `.xlsx`, `.xls`  
- 🔎 Uses **FAISS vector database** for fast similarity search  
- 🤖 Powered by **Groq LLM** for high-quality responses  
- 📝 Responses are returned in **structured, markdown-friendly format**  
- 🌐 Built with **FastAPI** backend + **HTML/CSS/JS frontend**  
- ⚡ Simple to deploy on **GitHub + Render/Vercel/Heroku**  

---

## 🛠️ Tech Stack
- **Backend:** FastAPI  
- **Frontend:** HTML + CSS + JavaScript  
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`  
- **Vector Database:** FAISS  
- **LLM Provider:** Groq (Llama 3 / Llama 4 Scout)  
- **Others:** LangChain, Pandas, dotenv  

---

## 📂 Project Structure
📦 rag-chatbot
┣ 📂 templates
┃ ┗ 📜 index.html # Frontend UI
┣ 📂 vector_db # Stores FAISS embeddings
┣ 📜 main.py # FastAPI backend (RAG logic)
┣ 📜 requirements.txt # Python dependencies
┣ 📜 README.md # Documentation
┗ 📜 .env # API keys (Groq) 


---

## 🚀 Setup & Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Add your API key
API-KEY=your_groq_api_key_here
5️⃣ Run the server
uvicorn app2:app --reload

💻 Usage

Open the app in your browser.

Upload a document (PDF/TXT/CSV/DOCX/XLSX).

Ask a question related to the uploaded document.

---

## 📸 Screenshots

### 🏠 Home Interface
![Home Interface](https://github.com/Upendra4204/Smart-Doc-Assistant/raw/main/Screenshot%202025-08-24%20130743.png)

### 💬 Chat Example
![Chat Example](https://github.com/Upendra4204/Smart-Doc-Assistant/raw/main/Screenshot%202025-08-24%20130805.png)




