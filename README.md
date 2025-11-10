# 🦙 Retrieval-Augmented Generation (RAG) Agent using LlamaIndex & Chroma

A modular, production-ready **Retrieval-Augmented Generation (RAG)** system built with [LlamaIndex](https://github.com/jerryjliu/llama_index), [Chroma](https://www.trychroma.com/), and both **OpenAI** and **Hugging Face** LLMs.

Upload PDFs, automatically build semantic indices, and query them in natural language — powered by dual embedding and inference pipelines for comparison, reliability, and flexibility.

---

## 🚀 Features

* 🔍 **Dual embedding modes** — OpenAI (`text-embedding-3-small`) and HuggingFace (`BAAI/bge-small-en-v1.5`)
* ⚙️ **Persistent vector store** with Chroma — reuses embeddings if the PDF is unchanged
* 🧠 **Smart semantic chunking** — automatic text segmentation using `SemanticSplitterNodeParser`
* 💬 **Dual LLM responses** — queries answered by both OpenAI (`gpt-3.5-turbo`) and HuggingFace Inference API (`Mixtral-8x7B-Instruct`)
* 🧾 **PDF-to-answer pipeline** — upload, embed, index, and chat — all in one command
* 🧩 **Evaluation support** — measures **Faithfulness**, **Relevancy**, and **Correctness** across engines
* 🧪 **Fully testable** — includes offline-safe unit tests (no API calls)
* ⚡ **Smart caching** — detects PDF content changes via hashing and reindexes automatically

---

## 🧰 Tech Stack

| Category     | Library                                                |
| ------------ | ------------------------------------------------------ |
| Framework    | [LlamaIndex](https://github.com/jerryjliu/llama_index) |
| Vector Store | [ChromaDB](https://www.trychroma.com/)                 |
| Embeddings   | OpenAI, HuggingFace (BGE-small)                        |
| LLMs         | OpenAI GPT-3.5 Turbo, HuggingFace Mixtral 8x7B         |
| Evaluation   | LlamaIndex evaluators                                  |
| Config       | python-dotenv                                          |
| PDF Parsing  | pypdf                                                  |
| Testing      | pytest + mocks                                         |
| Utilities    | pandas, tqdm, nest_asyncio                             |

---

## 📁 Project Structure

```
rag-agent-llamaindex/
├── src/
│   ├── __init__.py
│   ├── app.py                   # CLI entry point
│   ├── config.py                # Logging and environment setup
│   ├── loader.py                # PDF loader
│   ├── embedding_indexer.py     # Node splitting & embedding index creation
│   ├── rag_agent.py             # Dual query engine (OpenAI + HuggingFace)
│   └── evaluator.py             # Evaluation metrics & reporting
│
├── tests/
│   ├── __init__.py
│   └── test_rag_agent.py        # Offline-safe test suite
│
├── data/
│   └── sample.pdf               # Example document
│
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/rag-agent-llamaindex.git
cd rag-agent-llamaindex
```

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.\.venv\Scripts\activate       # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure your environment

Copy the example file and add your keys:

```bash
cp .env.example .env
```

Then fill it in:

```
OPENAI_API_KEY=sk-xxxx
HUGGINGFACE_API_KEY=hf_xxxx
```

If you don’t have API keys, the agent automatically falls back to **local HuggingFace embeddings**, allowing full offline functionality.

---

## ▶️ Usage

### Query a PDF

```bash
python -m src.app --pdf ./data/sample.pdf --query "Summarize this document"
```

**Output:**

```
✅ Existing vector indices found — skipping rebuild.

🧠 OpenAI → The document describes LIC’s New Jeevan Shanti policy with guaranteed annuity.
🤗 HuggingFace → This PDF outlines the annuity options and eligibility criteria under the plan.
```

### Evaluate performance

```bash
python -m src.app --pdf ./data/sample.pdf --run-eval
```

Example summary:

```
📊 EVALUATION RESULTS SUMMARY
----------------------------------------
🧠 OpenAI-based Index:
Faithfulness    : 88.0%
Relevancy       : 90.5%
Correctness     : 85.0%

🤗 HuggingFace-based Index:
Faithfulness    : 83.0%
Relevancy       : 88.0%
Correctness     : 81.0%
```

---

## 🧩 Offline Mode

No API keys? No problem.

When `OPENAI_API_KEY` isn’t set, the system automatically:

* Uses **BAAI/bge-small-en-v1.5** local embedding model
* Skips OpenAI calls entirely
* Runs queries fully offline

This makes development and testing seamless.

---

## 🧪 Testing

Run all tests:

```bash
pytest -v
```

Run only local tests:

```bash
pytest -v -k "split or index"
```

Run integration (live API) tests:

```bash
pytest -v --runlive
```

Offline tests create valid PDFs using `reportlab` and verify the full pipeline without hitting external APIs.

---

## 🧱 Example Dual Query Output

| Engine                                  | Response                                                                       |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| **OpenAI GPT-3.5 Turbo**                | “The document describes an annuity plan under LIC’s New Jeevan Shanti policy.” |
| **HuggingFace Mixtral (Inference API)** | “This PDF outlines LIC’s guaranteed pension product and key benefits.”         |

---

## 📊 Evaluation Metrics Explained

| Metric           | Meaning                                                |
| ---------------- | ------------------------------------------------------ |
| **Faithfulness** | Does the model’s answer align with the document facts? |
| **Relevancy**    | Is the answer focused on the question context?         |
| **Correctness**  | Does the model respond accurately and completely?      |

All evaluated using GPT-based evaluators via LlamaIndex.

---

## 🧠 Design Philosophy

This project follows **modular AI system design** principles:

* Clear separation between loading, embedding, retrieval, and generation.
* Environment-agnostic (works both with and without API keys).
* Fully reproducible pipeline for RAG benchmarking.
* Practical and inspectable code for AI engineers learning LlamaIndex.

---

## 🧹 Maintenance

| Command            | Purpose                  |
| ------------------ | ------------------------ |
| `black .`          | Auto-format code         |
| `flake8`           | Lint for PEP8 compliance |
| `pytest -v`        | Run all tests            |
| `pytest --cov=src` | Generate coverage report |
| `deactivate`       | Exit virtual environment |

---

## 🧾 License

This project is released under the [MIT License](LICENSE).

---

## 💡 Next Steps

* [ ] Add Streamlit UI for drag-and-drop PDF querying
* [ ] Integrate FAISS / Milvus for scalable multi-PDF retrieval
* [ ] Add caching layer (LangChain retriever or SQLite)
* [ ] Serve via FastAPI for production inference

