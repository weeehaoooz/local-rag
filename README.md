# Agentic Local RAG & Research Engine (LightRAG)

This project implements an **Agentic Local RAG** (Retrieval-Augmented Generation) system and **Deep Research Engine** that leverages Vector embeddings, Property Knowledge Graphs, and Web/Academic search for more accurate and context-aware information retrieval. It follows the **LightRAG** approach, combining granular entity-based retrieval with global community summarization, enhanced by an agentic orchestration layer.

It is built using **LlamaIndex** and **Ollama**, with **Neo4j** as the graph database backend.

## 🧠 The Agentic Approach

Unlike traditional "naive" RAG pipelines that execute a single, fixed `Embed -> Search -> Generate` sequence, this project implements a dynamic, autonomous **agentic architecture** capable of reasoning, planning, and self-correction. 

1. **Reasoning & Planning**: Complex, multi-topic user questions are not blindly searched. An LLM Decomposer analyzes the intent and breaks the prompt down into parallel, independent sub-queries.
2. **Tool-Calling Orchestration**: An intelligent Orchestrator acts as the "brain," evaluating each sub-query to dynamically dispatch specialized retrieval tools—autonomously routing between Local Vector Search, Neo4j Property Graph traversal, Web Search, or ArXiv Academic Search depending on the optimal data source.
3. **Query Transformation**: Utilizing conversational memory, Coreference Resolution, and HyDE (Hypothetical Document Embeddings), the system automatically expands vague queries into highly targeted search vectors without user intervention.
4. **Self-Reflection (Corrective RAG - CRAG)**: Retrieval isn't assumed to be successful. An Evaluator agent grades the retrieved context. If relevant information is missing, the system initiates a self-correction loop: rewriting the query and autonomously executing web or deep research to fill knowledge gaps.
5. **Anti-Hallucination Guardrails**: Before output, the final generated response is fact-checked by the Evaluator against the retrieved context to guarantee groundedness and flag fabricated claims.
6. **State Streaming**: The multi-step agentic workflow (e.g., *"Decomposing query"*, *"Grading context"*, *"Searching web"*) is streamed live to the UI, providing complete transparency into the system's thought process.

---

## 💡 The LightRAG Paradigm

While the agentic orchestrator acts as the system's "brain," the underlying data structure and retrieval mechanics are driven by the **LightRAG** approach. This solves the limitations of both pure vector search (which struggles with holistic understanding) and traditional GraphRAG (which can be computationally expensive to build).

- **Dual-Level Search**: The system natively supports two paradigms of knowledge graph traversal:
  - **Local Retrieval**: Extracts specific, granular entities and their direct topological relationships (N-hop neighbors) to answer highly specific questions.
  - **Global Retrieval**: Utilizes graph community detection to cluster related entities, generating pre-computed high-level summaries. This enables the system to answer broad, overarching questions that span across multiple documents.
- **Incremental Knowledge Building**: Unlike traditional pipelines that require a full rebuild whenever new data is added, our implementation uses hash-tracking to seamlessly append only *new* entities and relationships to the existing graph.
- **Efficient Extraction**: By filtering and optimizing the extracted graph topology, it maintains high retrieval accuracy without the exponential token costs of standard graph generation.

---

## 🗂️ The Indexing Flow & Considerations

To ensure high-quality graph and vector retrieval, the data ingestion pipeline handles structural nuances *before* they reach the databases:

- **Semantic Chunking for Coherence**: Instead of basic character-count splitting, the system evaluates logical sentence boundaries and topic shifts. This ensures that parent-child hierarchical chunks ("Small-to-Big" chunking strategy) retain their underlying meaning and prevent data fragmentation.
- **Automated Coreference Resolution**: An LLM preprocessing step dynamically resolves ambiguous pronouns in the text (e.g., changing "it" to "the LlamaIndex framework" across paragraphs). This strictly prevents dangling or disconnected entities in the Neo4j Knowledge Graph.
- **Vision-Based Metadata Extraction**: Before chunking, layout-heavy documents (like PDFs with tables or diagrams) are parsed to extract structural metadata, preserving spatial and tabular context.
- **Actor-Critic KG Extraction**: During graph creation, an LLM extracts entities and relationships, while an independent "critic" loop fact-checks each generated triplet against the source text to block hallucinations.
- **Incremental Indexing via Hashing**: The coordinator service tracks file hashes. When processing directories, it skips untouched files, seamlessly updating only modified or newly ingested documents.

---

## 🚀 Features

### Core RAG Engine
- **Dual-Level Retrieval**: Combines low-level relational context (N-hop neighbors) with high-level community summaries.
- **Hierarchical Chunking**: Uses a Small-to-Big strategy (Semantic Splitter for parents, Sentence Splitter for children) for precise extraction.
- **Actor-Critic Extraction**: Fact-checks Knowledge Graph triplets against the source text to ensure accuracy.
- **Local-First**: All LLM processing (**Gemma 4**) and embeddings (**Nomic Embed**) run locally via Ollama.
- **Incremental Updates**: Uses hash-based tracking to update the graph without full re-indexing.

### Agentic Orchestration
- **Query Transformation**: Implements HyDE (Hypothetical Document Embeddings) and Multi-Query decomposition for complex reasoning.
- **Self-Reflection (CRAG)**: Grading retrieval results and answer groundedness to detect and correct hallucinations.
- **Intelligent Routing**: Dynamic planning and tool selection between Vector, Graph, and Web/Academic sources.
- **Query Decomposition**: Splits multi-topic questions into independent sub-queries for parallel retrieval.

### Deep Research & Web Integration
- **Hybrid Search**: Integrates DuckDuckGo (Web) and ArXiv (Academic) research tools.
- **Deep Research Mode**: Automated terminology discovery and trafilatura-based article scraping for real-time document ingestion.
- **Conversational Research**: Interactive multi-turn research workflows for in-depth topic exploration.

### Data & Evaluation
- **Structured Data Support**: Native handling for CSV, JSON, and SQL datasets with table-aware indexing.
- **Evaluation Framework**: Built-in RAG validation metrics (Precision, Recall, Faithfulness, Relevance) to measure system performance iterations.

## 🛠️ Tech Stack

- **Framework**: [LlamaIndex](https://www.llamaindex.ai/)
- **LLM**: [Ollama](https://ollama.com/) (Gemma 4 / Llama 3)
- **Embeddings**: Ollama (Nomic Embed Text)
- **Graph Database**: [Neo4j](https://neo4j.com/)
- **Utilities**: [Trafilatura](https://trafilatura.readthedocs.io/) (Scraping), DuckDuckGo-Search (Web), ArXiv (Academic Research)
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
- **[backend/retrieval/services/](backend/retrieval/services/)**: Agentic services (Orchestrator, Decomposer, Transformer, Evaluator).
- **[backend/research/](backend/research/)**: Deep Research engine with search tools and scraper.
- **[backend/scripts/](backend/scripts/)**: Management utilities for graph cleanup and status checks.

## 🚀 Getting Started

Follow these steps to get your local KG-RAG system up and running.

### 1. Project Setup
Clone the repository and create a Python virtual environment:
```bash
# Create and activate virtual environment
python3 -m venv backend/venv
source backend/venv/bin/activate  # On Windows: backend\Scripts\activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Verify Your Environment
Before indexing, ensure your local services (Ollama and Neo4j) are reachable:
```bash
python backend/scripts/check_connections.py
```
*This will check for required models (Gemma 4, Nomic Embed) and Neo4j connectivity.*

### 3. Index Your Data
1. Place your `.pdf`, `.docx`, or `.txt` files in `backend/data/`.
2. Run the initial indexing scan:
   ```bash
   # This will populate the Vector and Property Graph indexes
   python backend/scripts/manage_indexes.py stats
   ```
   *Note: On first run, it will automatically trigger the ingestion of new files.*
3. **Critical**: Run community clustering to enable "Global" retrieval:
   ```bash
   python backend/scripts/manage_indexes.py cluster --summarize
   ```

### 4. Launch the Application

#### Option A: Web Interface (Recommended)
The full experience includes a FastAPI backend and an Angular frontend.

**Start the Backend API:**
```bash
# Runs on http://localhost:8000
python backend/api.py
```

**Start the Frontend UI:**
```bash
cd frontend
npm install
npm start  # Runs on http://localhost:4200
```

#### Option B: Interactive CLI
For a pure terminal-based chat experience:
```bash
python backend/main.py
```

#### Option C: Research Engine
Launch the research module to search Web or Academic (ArXiv) sources:
```bash
# This will enter an interactive session for Deep Research
python backend/scripts/research.py
```

## 🔍 How it Works

1. **Ingestion & Hierarchical Chunking**: Documents (PDF, DOCX, TXT, CSV, JSON) are loaded via layout-aware loaders and split into "Small-to-Big" chunks for balanced context and precision.
2. **Actor-Critic KG Extraction**: Entities and relationships are extracted using an LLM loop that fact-checks every triplet against the source text to prevent hallucinations.
3. **Agentic Pipeline**: Every query is processed through a multi-stage workflow:
   - **Transformer**: Re-writes queries and generates hypothetical documents (HyDE) to align with vector space.
   - **Decomposer**: Breaks complex questions into simpler sub-queries.
   - **Orchestrator**: Dynamically selects the best tools (Vector, Graph, or Web Search) for each sub-query.
4. **Self-Reflection (CRAG)**: Retrieval results are graded on relevance; if the context is insufficient, the system automatically triggers a web search to fill gaps. The final answer is then grounded against the retrieved documents.

## 🤝 Contributing

This is a research/testing project for local RAG implementations. Feel free to open issues or submit PRs for improvements!
