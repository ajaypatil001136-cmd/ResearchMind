# 🧠 ResearchMind — AI Research Assistant

**A local PDF-based Retrieval-Augmented Generation (RAG) application that helps users understand and explore research documents using natural language questions.**

## Overview

ResearchMind is an AI-powered research assistant built with Python and Streamlit. It allows users to upload PDF documents, extract their content, and ask questions about the information inside them.

It uses vector embeddings and FAISS similarity search to retrieve relevant document passages, then uses a locally running Large Language Model (LLM) through Ollama to generate context-aware answers.

## Key Features

- Upload and process PDF documents.
- Extract text from PDF pages.
- Split documents into smaller text chunks.
- Generate semantic embeddings.
- Store and search embeddings using FAISS.
- Retrieve relevant passages for user questions.
- Generate answers using a local Ollama model.
- Display source documents and page references.
- Interactive Streamlit interface.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Frontend | Streamlit |
| PDF processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database/index | FAISS |
| Local LLM runtime | Ollama |
| Language model | llama3.2:3b |

## Architecture

1. **PDF Upload:** User uploads research documents.
2. **Text Extraction:** PyMuPDF extracts page-wise text.
3. **Chunking:** Extracted text is divided into smaller passages.
4. **Embedding Generation:** Sentence Transformers converts passages into vectors.
5. **Vector Indexing:** FAISS stores vectors for similarity search.
6. **Retrieval:** Relevant passages are retrieved for a question.
7. **Answer Generation:** Ollama generates a response using retrieved context.
8. **Source Display:** The application presents supporting document references.

## Project Structure

```text
ResearchMind/
│
├── app.py
├── chunker.py
├── embedder.py
├── vector_store.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ResearchMind
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Ollama

Install Ollama separately from:

https://ollama.com/

Pull the model:

```bash
ollama pull llama3.2:3b
```

Ensure Ollama is running before launching the application.

### 6. Run ResearchMind

```bash
streamlit run app.py
```

## How It Works

ResearchMind follows a Retrieval-Augmented Generation (RAG) approach.

Instead of asking the language model to answer questions using only its pretrained knowledge, the application first retrieves relevant passages from the uploaded documents and supplies those passages as context to the model.

This helps make responses more relevant to the documents being explored. However, answer accuracy and grounding still need to be evaluated.

## Current Limitations

- Answer quality depends on PDF text extraction and retrieval quality.
- Scanned PDFs may require OCR support.
- Local model performance depends on available system resources.
- Formal retrieval accuracy and latency benchmarks are not yet included.

## Future Improvements

- OCR support for scanned documents.
- Improved retrieval and reranking.
- Formal RAG evaluation.
- Persistent vector storage.
- More advanced document management.
- Support for additional local language models.

## Author

**Developed as an AI/ML engineering portfolio project.**

Built to explore Retrieval-Augmented Generation, semantic search, vector databases, and local LLM deployment.
