# VectorDB — Build a Vector Database from Scratch in Python + JavaScript

A fully working **Vector Database** built from scratch in **Python (FastAPI)** with a modular **vanilla JavaScript** web UI.
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.

> Built as an educational project to show how production vector databases like Pinecone, Weaviate, and Chroma actually work under the hood.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info (+ auto docs at `/docs`) |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)                ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, achieving O(log N) complexity instead of O(N) for brute force.

---

## Prerequisites

1. **Python 3.10+** — https://www.python.org/downloads/
2. **Ollama** (runs the local AI models) — https://ollama.com

```powershell
ollama pull nomic-embed-text   # ~274 MB embedding model
ollama pull llama3.2           # ~2 GB language model
```

> **Minimum specs for Ollama:** 8GB RAM recommended. The models will use ~3GB total.
> The demo-vector features work without Ollama; only the Documents / Ask AI tabs need it.

---

## Quick Start

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

You should see:
```
=== VectorDB Engine (Python) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open **http://localhost:8080** for the UI, or **http://localhost:8080/docs** for interactive API docs.

### Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8080` | HTTP port |
| `HOST` | `0.0.0.0` | Bind address |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `GEN_MODEL` | `llama3.2` | Generation model |

Example — use the faster 1B model:
```powershell
$env:GEN_MODEL = "llama3.2:1b"; python run.py
```

### Running Tests

```powershell
cd backend
pip install -r requirements-dev.txt
python -m pytest -q
```

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Type any concept in the search box: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose your algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear with distances, the matching point glows on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to run all 3 algorithms and compare their speed

### Tab 2: Documents (Real Embeddings)

1. Type a title (e.g., `Operating Systems Notes`)
2. Paste any text — lecture notes, textbook paragraphs, Wikipedia articles
3. Click **⚡ EMBED & INSERT**
4. Long documents are automatically split into overlapping 250-word chunks
5. Each chunk gets its own embedding and is stored in a separate HNSW index

### Tab 3: Ask AI (RAG Pipeline)

1. Insert some documents in Tab 2 first
2. Type a question and click **🤖 ASK AI** (or Ctrl+Enter)

```
1. Your question → embedded with nomic-embed-text (768D vector)
2. HNSW search → finds the most semantically similar chunks
3. Retrieved chunks → sent as context to llama3.2
4. llama3.2 → generates an answer based on your documents
```

---

## REST API Reference

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector `{"metadata","category","embedding"}` |
| `DELETE` | `/delete/{id}` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/{id}` | — | Delete document chunk |
| `POST` | `/doc/search` | `{"question":"...","k":3}` | Retrieval only |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status and model info |

Errors are returned as `{"error": "..."}` with an appropriate HTTP status code.

---

## Project Structure

```
Your-OWN-AI-main/
├── backend/                         ← Python (FastAPI)
│   ├── run.py                       ← entry point
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                  ← app factory, CORS, static frontend
│   │   ├── config.py                ← settings (env-overridable)
│   │   ├── core/
│   │   │   ├── models.py            ← VectorItem, DocItem
│   │   │   └── distance.py          ← euclidean / cosine / manhattan
│   │   ├── indexes/
│   │   │   ├── brute_force.py       ← O(N·d) exact
│   │   │   ├── kd_tree.py           ← exact, axis-aligned partitioning
│   │   │   └── hnsw.py              ← approximate, multilayer graph
│   │   ├── db/
│   │   │   ├── vector_db.py         ← unified 16D demo DB over all 3 indexes
│   │   │   └── document_db.py       ← HNSW over real Ollama embeddings
│   │   ├── services/
│   │   │   ├── ollama_client.py     ← async /api/embeddings + /api/generate
│   │   │   ├── chunker.py           ← overlapping word windows
│   │   │   ├── rag.py               ← ingest / retrieve / ask pipeline
│   │   │   └── demo_data.py         ← 20 seed vectors
│   │   └── api/
│   │       ├── deps.py              ← DI container + Depends() getters
│   │       ├── schemas.py           ← Pydantic request models
│   │       └── routes/              ← vectors.py, documents.py, system.py
│   └── tests/test_vectordb.py
│
└── frontend/                        ← Vanilla JS (ES modules) + CSS
    ├── index.html
    ├── css/                         ← tokens, base, layout, components, search, documents
    └── js/
        ├── main.js                  ← boot + event wiring
        ├── config.js  state.js      ← constants, shared state + event bus
        ├── services/api.js          ← all fetch calls
        ├── lib/                     ← embedding.js, pca.js (pure logic)
        ├── components/              ← scatterPlot.js, vectorChart.js
        ├── features/                ← tabs, search, benchmark, documents, rag
        └── utils/dom.js             ← $, escapeHtml, withBusy
```

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

Nodes are inserted into a multilayer graph. Each node randomly gets assigned a maximum layer. Layer 0 has all nodes with many connections; higher layers have fewer nodes (exponentially fewer) with longer-range connections.

**Insert:** Start at the top layer, greedily find the nearest node, drop a layer, repeat. At each layer from your assigned max down to 0, run a beam search (ef_construction=200) and connect to the M nearest neighbors bidirectionally.

**Search:** Same greedy descent from top layer. At layer 0, expand to ef nearest candidates using a priority queue.

**Why it's fast:** The upper layers act like a highway — you quickly get to the right neighborhood, then zoom in at layer 0.

### KD-Tree (K-Dimensional Tree)

Binary space partitioning. Each node splits space along one dimension (cycling through all dimensions). Search prunes entire subtrees when the closest possible point in that subtree can't beat the current best.

**Weakness:** Degrades with high dimensions (curse of dimensionality). Works well for ≤20D, becomes close to brute force at 768D.

### Why HNSW Wins at High Dimensions

KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions, almost all the space is near the boundary of the hypersphere — no subtrees get pruned. HNSW's graph-based approach doesn't have this problem.

---

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in header | Run `ollama serve` in a terminal |
| Embedding takes forever | Ollama is downloading the model on first use, wait 2 min |
| Port 8080 already in use | `$env:PORT=8081; python run.py`, or kill it: `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F` |
| LLM answer is slow | Normal — llama3.2 takes 10–30s on a laptop CPU. Use `GEN_MODEL=llama3.2:1b` |

---

## License

MIT — use this however you want.
