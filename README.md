# 🧠 ResearchMind — Local AI Research Assistant

**ResearchMind is a local, PDF-based Retrieval-Augmented Generation (RAG) application that lets you ask natural-language questions about research papers and inspect the passages used to answer them.**

## ✨ Features

- Upload one or multiple PDF research papers.
- Extract page-level text using PyMuPDF.
- Split text into overlapping chunks.
- Generate semantic embeddings using FastEmbed.
- Search relevant passages using FAISS.
- Generate answers using a locally running Ollama model.
- Display source filenames, page numbers, retrieved passages and similarity scores.
- Interactive Streamlit dashboard.

## 🖥️ Application Screenshots

### Dashboard
![ResearchMind Dashboard](screenshots/dashboard.png)

### Uploaded Documents
![Document Upload](screenshots/document-upload.png)

### RAG Answer and Source Evidence
![RAG Answer](screenshots/rag-answer.png)

## 🧰 Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Interface | Streamlit |
| PDF Processing | PyMuPDF |
| Embeddings | FastEmbed |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Local LLM Runtime | Ollama |
| Language Model | llama3.2:3b |

## 🏗️ Architecture

```text
PDF Upload
    ↓
PDF Text Extraction (PyMuPDF)
    ↓
Text Chunking
    ↓
Embedding Generation (FastEmbed)
    ↓
Vector Indexing (FAISS)
    ↓
User Question
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Relevant Context Retrieval
    ↓
Local LLM (Ollama)
    ↓
Generated Answer + Source Evidence
```

## 📁 Project Structure

```text
ResearchMind/
├── app.py
├── chunker.py
├── embedder.py
├── vector_store.py
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
    ├── dashboard.png
    ├── document-upload.png
    └── rag-answer.png
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ajaypatil001136-cmd/ResearchMind.git
cd ResearchMind
```

### 2. Create a virtual environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama from https://ollama.com/

Download the model:

```bash
ollama pull llama3.2:3b
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

http://localhost:8501

## 🔍 How It Works

ResearchMind follows a Retrieval-Augmented Generation workflow:

1. Extracts text from uploaded PDFs.
2. Converts text into overlapping chunks.
3. Generates vector embeddings.
4. Retrieves relevant passages using FAISS similarity search.
5. Supplies retrieved context to a local LLM.
6. Displays the generated answer with source evidence.

## ⚠️ Current Limitations

- Answer quality depends on extraction, retrieval and model performance.
- Scanned PDFs may require OCR support.
- Similarity scores are not guarantees of answer correctness.
- Persistent vector storage is not implemented.
- Formal retrieval accuracy and latency benchmarks are not yet available.

## 🚀 Future Improvements

- OCR support for scanned documents.
- Improved retrieval and reranking.
- Formal RAG evaluation.
- Persistent vector storage.
- Automated testing and performance monitoring.

## 👨‍💻 About

ResearchMind is an AI/ML engineering portfolio project exploring Retrieval-Augmented Generation, semantic search, vector databases, PDF processing and local LLM integration.
