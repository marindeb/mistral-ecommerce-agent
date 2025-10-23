# Mistral E-commerce Agent

A RAG + Agent system powered by Mistral 7B via Ollama and ChromaDB, simulating an e-commerce assistant able to query internal data and suggest insights.

## Features
- 🔎 Retrieval-Augmented Generation (RAG) on internal docs
- 🧮 Agent tools for product analytics (via Pandas)
- ⚙️ FastAPI backend + Streamlit UI
- 🧰 Dockerized & CI-ready

## Stack
- Python 3.10+
- LangChain, ChromaDB
- Mistral via Ollama
- FastAPI, Streamlit
- Docker, GitHub Actions

## Next steps
1. Generate synthetic e-commerce data (`notebooks/generate_data.ipynb`)
2. Build the RAG pipeline (`app/rag_pipeline.py`)
3. Add agent tools (`app/tools.py`)
4. Run locally via Docker Compose
