# VX-RAG

A hybrid RAG (Retrieval-Augmented Generation) + MCP (Model Context Protocol) system for processing and querying a large document corpus from vx-underground. The system supports CPU-only inference for embeddings and provides a modular architecture for scalable document processing.

## Architecture Overview

### RAG Component

- **Ingestion**: Loads and preprocesses PDF, TXT, and Markdown files from the vx-underground corpus (~10,000+ documents).
- **Indexing**: Creates vector embeddings using local models (sentence-transformers or nomic-embed-text) and stores them in ChromaDB or FAISS.
- **Querying**: Retrieves relevant document chunks based on user queries and provides context for generation.

### MCP Component

- **Bridge**: Acts as an intermediary between external LLMs (GPT, Claude, Mistral) and the RAG engine.
- **Routes**: Provides REST API endpoints for MCP communication.
- **Auth**: Handles authentication and authorization for secure access.

The MCP layer manages context flow: external LLMs send queries to MCP, which retrieves relevant context from RAG and returns it for generation.

## Project Structure

```text
VX-RAG/
│
├── src/
│   ├── rag/
│   │   ├── ingest.py      # Document loading and preprocessing
│   │   ├── index.py       # Vector index creation with LlamaIndex/ChromaDB
│   │   ├── query.py       # Query handling and response formatting
│   │   ├── embeddings.py  # Local embedding generation
│   │   ├── config.py      # Configuration management
│   │   └── utils.py       # Helper functions
│   │
│   ├── mcp/
│   │   ├── bridge.py      # MCP-RAG bridge logic
│   │   ├── routes.py      # Flask API routes
│   │   └── auth.py        # Authentication/authorization
│   │
│   └── cli.py             # Command-line interface
│
├── data/
│   ├── raw/
│   │   ├── md/            # Raw Markdown files
│   │   ├── pdf/           # Raw PDF files
│   │   └── txt/           # Raw text files
│   ├── processed/         # Preprocessed text documents
│   └── index/             # Vector index storage
│
├── tests/
│   ├── test_ingest.py     # Ingestion tests
│   ├── test_query.py      # Query tests
│   └── test_bridge.py     # MCP bridge tests
│
├── config/
│   └── settings.yaml      # Configuration file
│
├── Dockerfile             # Containerization
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

### Local Setup

1. **Clone and setup environment:**

   ```bash
   git clone https://github.com/Aletheia-Praxis/VX-RAG.git
   cd vx-rag
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure settings:**
   Edit `config/settings.yaml` to set paths, model names, and parameters.

3. **Download models (optional):**
   The system will automatically download embedding models on first use.

### Docker Setup

```bash
docker build -t vx-rag .
docker run -p 5000:5000 vx-rag
```

## Usage

### Data Ingestion

Place your documents in `data/raw/md/`, `data/raw/pdf/`, `data/raw/txt/`.

Run ingestion:

```bash
python src/cli.py ingest --data-dir data/raw
```

### Index Creation

Create the vector index:

```bash
python src/cli.py index --persist-dir data/index
```

### Local Querying

Query the system via CLI:

```bash
python src/cli.py query "What is malware analysis?"
```

### MCP API Usage

Start the MCP server:

```bash
python src/mcp/routes.py
```

The API will be available at `http://localhost:5000`.

Example API call:

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain VX Underground"}'
```

### Connecting External LLMs

Configure your LLM client to use the MCP API endpoint. The MCP bridge will:

1. Receive queries from the LLM
2. Retrieve relevant context from RAG
3. Return context for the LLM to generate responses

Set environment variables:

```bash
export MCP_API_KEY=your_api_key
export LLM_API_KEY=your_llm_key
```

## Configuration

Edit `config/settings.yaml`:

```yaml
data_dir: "./data"
index_dir: "./data/index"
embedding_model: "..."
chunk_size: 512
vector_store: "chromadb"  # or "faiss"
```

## Testing

Run tests:

```bash
pytest tests/
```

## Requirements

- Python 3.11+
- Disk space: ~20GB for index (depending on corpus size)
- Brains and understanding what you are doing

## License

The **source code** for this project is licensed under the [MIT](./LICENSE) license.

**Note:** The *dataset* consisting of the raw text files (`.md`, `.pdf`, `.txt`) obtained from [vx-underground.org](https://vx-underground.org/) is distributed under **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.
These materials are provided strictly for **educational and research purposes**, and **commercial use is prohibited**.

See the [DATA LICENSE](./DATA_LICENSE) file for details.
[Official License Text](https://creativecommons.org/licenses/by-nc/4.0/legalcode)
