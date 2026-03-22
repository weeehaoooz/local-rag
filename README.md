# Local Hybrid RAG with Knowledge Graphs

This project implements a local RAG (Retrieval-Augmented Generation) system that leverages both Vector embeddings and Property Knowledge Graphs for more accurate and context-aware information retrieval. It is built using **LlamaIndex** and **Ollama**, with **Neo4j** as the graph database backend.

## 🚀 Features

- **Hybrid Retrieval**: Combines relational context from a Knowledge Graph with structural summaries and vector search.
- **Schema-Guided Extraction**: Automatically generates and optimizes "guardrails" (schemas) per document category for more precise entity and relationship extraction.
- **Local-First**: All LLM processing (Llama 3) and embeddings (Nomic Embed) run locally via Ollama.
- **Knowledge Graph Management**: Includes tools for indexing, cleaning (deduplication/merging), and clearing the Neo4j graph.
- **Persistence**: Tracks file changes and guardrail updates to avoid redundant re-indexing.

## 🛠️ Tech Stack

- **Framework**: [LlamaIndex](https://www.llamaindex.ai/)
- **LLM**: [Ollama](https://ollama.com/) (Llama 3)
- **Embeddings**: Ollama (Nomic Embed Text)
- **Graph Database**: [Neo4j](https://neo4j.com/)
- **Environment**: Python 3.x

## 📋 Prerequisites

1. **Ollama**: Install and run Ollama. Pull the required models:
   ```bash
   ollama pull llama3
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

- `indexer.py`: The main entry point for document indexing. Handles categorization, guardrail generation, and KG construction.
- `main.py`: Interactive CLI for querying the RAG system.
- `engine.py`: The hybrid retrieval engine that orchestrates context gathering from various indices.
- `indexers/`: Modular indexers (Vector, Summary, Graph) and guardrail management.
- `clean_graph.py`: Script to merge similar entities and relationships in the KG.
- `clear_knowledge_graph.py`: Utility to wipe the Neo4j database.
- `data/`: Directory where your source documents (.txt, .pdf) should be placed.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Index Your Data
Place your documents in the `./data` folder and run the indexer:
```bash
python indexer.py --clean
```
*Options:*
- `--force`: Re-index all files regardless of changes.
- `--clean`: Run graph cleanup after indexing.
- `--passes N`: Number of extraction passes for the graph (default: 1).
- `--hybrid`: Enable hybrid (schema + free-form) extraction.

### 3. Start Chatting
```bash
python main.py
```

## 🔍 How it Works

1. **Categorization**: Documents are automatically categorized based on their folder structure.
2. **Guardrails**: For each category, Llama 3 generates a specific schema (entities and relationship types) to guide the KG extraction process.
3. **Index Stages**:
   - **Vector Index**: For semantic search.
   - **Summary Index**: For high-level document understanding.
   - **Property Graph Index**: For mapping complex relationships between entities.
4. **Hybrid Querying**: When you ask a question, the `HybridEngine` retrieves relevant sub-graphs from Neo4j and structural summaries from the document index to provide a comprehensive context to the LLM.

## 🤝 Contributing

This is a research/testing project for local RAG implementations. Feel free to open issues or submit PRs for improvements!
