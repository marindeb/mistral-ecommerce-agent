# Mistral E-commerce Agent

A production-grade RAG + Agent system powered by Mistral 7B (via Ollama) and ChromaDB.
It simulates an internal e-commerce assistant able to query operational data and generate insights such as delivery performance, product returns, and category-level analytics.

---

## Overview

This project demonstrates how to design, deploy, and document an end-to-end LLM product:
- **Retrieval-Augmented Generation (RAG)** for grounding model responses in internal product data.
- **Pandas Agent** for dynamic analytics and reasoning over structured datasets.
- **FastAPI backend** for unified API access and orchestration.
- **Streamlit UI (optional)** for interactive exploration and demos.

---

## Tech Stack

- **Language:** Python 3.10.13
- **LLM:** Mistral 7B via Ollama
- **Frameworks:** LangChain, ChromaDB, FastAPI, Streamlit
- **Infrastructure:** Docker, GitHub Actions

---

## Architecture & Design Rationale

### RAG Pipeline
The RAG component retrieves semantically similar text chunks from internal product data
and uses them to ground Mistral's responses.
Chunking improves retrieval granularity and prevents context overflow, but may also fragment related information across documents — requiring careful tuning of chunk size and overlap.

### Agent Layer
The Pandas agent complements the RAG pipeline by allowing the LLM to reason dynamically
over structured datasets (e.g., delivery performance, return rates).
It generates and executes Pandas code on demand, making it suitable for exploratory analytics and ad-hoc data queries.

### Backend API
A FastAPI service exposes unified endpoints (`/query`, `/health`, etc.), integrating both the RAG and agent layers under a consistent interface for frontend or external use.

---

## Project Structure

```
mistral-ecommerce-agent/
├── app/              # Core logic: RAG, Agent, constants
├── data/             # Synthetic datasets and Chroma index
├── notebooks/        # Data generation
├── tests/            # Unit and integration tests
├── ui/               # (Optional) Streamlit interface
└── requirements*.txt # Dependencies (base + dev)
```

---

## Environment Setup (pyenv + pyenv-virtualenv)

Ensure you have [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) installed.

```bash
# From the project root
pyenv install 3.10.13          # if not already installed
pyenv virtualenv 3.10.13 mistral-agent-env
pyenv local mistral-agent-env

# Activate environment and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Development Setup

For development:
```bash
pip install -r requirements-dev.txt
pre-commit install
```

The pre-commit hook runs `ruff` automatically to ensure consistent formatting
and import order before each commit.

---

## Running Ollama

Ollama must be running locally for the Mistral model to respond.

```bash
# Start the Ollama server
ollama serve

# (optional) Run it in the background
ollama serve &

# Pull the Mistral model
ollama pull mistral
```

Check that it is active:
```bash
curl http://localhost:11434/api/tags
```

---

## Usage

1. Generate synthetic e-commerce data:
   ```bash
   jupyter notebook notebooks/generate_data.ipynb
   ```
2. Build and persist the RAG index:
   ```bash
   python -m app.rag_pipeline
   ```
3. Launch the FastAPI backend:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Optionally, interact via Streamlit UI.

API documentation is available at:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Running Tests

Unit and integration tests are located in the `tests/` folder.
Run all tests with:

```bash
pytest -v
```

Pytest is configured to recognize the `app/` package via `pytest.ini`.

---

## Manual Component Tests

**RAG pipeline**
```bash
python -m app.rag_pipeline
```

**Pandas agent**
```bash
python -m app.agent
```

Example:
```python
from app.agent import ask_agent
ask_agent("Which categories have the highest return rate?")
```

---

## Next Steps

- Add lightweight confidence scoring and response logging.
- Integrate guardrails (column access restriction, sandboxed execution).
- Package and deploy using Docker Compose.
- Build a Streamlit UI for demonstration.
- Add Dockerized testing (CI/CD ready).
