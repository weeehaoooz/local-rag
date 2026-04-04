# Local Hybrid RAG with Knowledge Graphs (LightRAG)

This project implements a local RAG (Retrieval-Augmented Generation) system that leverages both Vector embeddings and Property Knowledge Graphs for more accurate and context-aware information retrieval. It follows the **LightRAG** approach, combining granular entity-based retrieval with global community summarization.

It is built using **LlamaIndex** and **Ollama**, with **Neo4j** as the graph database backend.

## 🚀 Features

- **Dual-Level Retrieval**: Combines low-level relational context (N-hop neighbors) with high-level community summaries.
- **Hierarchical Chunking**: Uses a Small-to-Big strategy (Semantic Splitter for parents, Sentence Splitter for children) for precise extraction.
- **Actor-Critic Extraction**: Fact-checks Knowledge Graph triplets against the source text to ensure accuracy.
- **Local-First**: All LLM processing (**Gemma 4**) and embeddings (**Nomic Embed**) run locally via Ollama.
- **Incremental Updates**: Uses hash-based tracking to update the graph without full re-indexing.

## 🛠️ Tech Stack

- **Framework**: [LlamaIndex](https://www.llamaindex.ai/)
- **LLM**: [Ollama](https://ollama.com/) (Gemma 4)
- **Embeddings**: Ollama (Nomic Embed Text)
- **Graph Database**: [Neo4j](https://neo4j.com/)
- **Environment**: Python 3.x

## 📋 Prerequisites

1. **Ollama**: Install and run Ollama. Pull the required models:
   ```bash
   ollama pull gemma4
   ollama pull nomic-embed-text
   ```
2. **Neo4j**: Have a running Neo4j instance (Local Desktop or Docker).
3. **Environment**: Create a `.env` file in the root directory (see [Configuration](#-configuration)).

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
OLLAMA_BASE_URL=http://localhost:11434
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

## 📂 Project Structure

- **[docs/technical_specs.md](docs/technical_specs.md)**: Detailed technical architecture and specifications.
- **[backend/main.py](backend/main.py)**: Interactive CLI for querying the RAG system.
- **[backend/config.py](backend/config.py)**: Centralized configuration for models, storage, and database.
- **[backend/ingestion/](backend/ingestion/)**: Layout-aware loaders and coreference preprocessors.
- **[backend/indexing/](backend/indexing/)**: Graph extraction, community detection, and vector indexing.
- **[backend/retrieval/](backend/retrieval/)**: Hybrid retrieval engine with local/global routing logic.
- **[backend/scripts/](backend/scripts/)**: Management utilities for graph cleanup and status checks.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Index Your Data
Place your documents in the `./backend/data` folder and start the background ingestion service or run a manual scan:
```bash
# Example manual scan if integrated into a script
python backend/scripts/manage_indexes.py stats
```

### 3. Start Chatting
```bash
python backend/main.py
```

## 🤝 Contributing

This is a research/testing project for local RAG implementations. Feel free to open issues or submit PRs for improvements!
