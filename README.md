# VX-RAG

A hybrid RAG (Retrieval-Augmented Generation) + MCP (Model Context Protocol) system for processing and querying a large document corpus from vx-underground. The system supports CPU-only inference for embeddings and provides a modular architecture for scalable document processing.

## Architecture Overview

### RAG Component

- **Ingestion**: Loads and preprocesses PDF, TXT, and Markdown files from the vx-underground corpus (~10,000+ documents).
- **Indexing**: Creates vector embeddings using local models (sentence-transformers or nomic-embed-text) and stores them in FAISS.
- **Querying**: Retrieves relevant document chunks based on user queries and provides context for generation.

### MCP Component

- **Server**: FastMCP-based MCP server providing tools and resources for LLM integration.
- **Tools**: Query tool for document retrieval and response generation.
- **Resources**: Health status and system context endpoints.
- **Middleware**: Rate limiting, request queuing, structured logging, and performance metrics.

The MCP server enables direct integration with IDEs and LLMs through the Model Context Protocol, allowing tools to query the RAG system for contextual information. The middleware layer ensures stable operation with automatic rate limiting (2 concurrent requests by default) and request queuing (10 requests maximum).

## Project Structure

```bash
VX-RAG/
│
├── src/
│   ├── rag/
│   │   ├── ingest.py      # Document loading and preprocessing
│   │   ├── build_index.py # Vector index creation with FAISS
│   │   ├── query.py       # Query handling and response formatting
│   │   ├── embeddings.py  # Local embedding generation
│   │   ├── config.py      # Configuration management
│   │   └── utils.py       # Helper functions
│   │
│   ├── mcp/
│   │   ├── server.py      # FastMCP server with tools and resources
│   │   ├── bridge.py      # MCP-RAG bridge logic
│   │   ├── routes.py      # Additional MCP routes
│   │   └── auth.py        # Authentication/authorization
│   │
│   └── cli.py             # Command-line interface
│
├── data/
│   ├── raw/
│   │   ├── pdf/           # Raw PDF files
│   │   └── txt/           # Raw text files (TODO: MD support)
│   ├── processed/         # Preprocessed text documents
│   └── index/             # FAISS vector index storage
│
├── tests/
│   ├── test_ingest.py     # Ingestion tests
│   ├── test_query.py      # Query tests
│   └── test_server.py     # MCP server tests
│
├── requirements.txt       # Python dependencies
├── README.md
└── LICENSE
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

### Incremental Index Updates

Add new documents to existing index without full rebuild:

```bash
python src/cli.py update-index --data-dir data/raw --persist-dir data/index
```

This follows the technical standard for incremental FAISS updates, automatically creating backups and maintaining index integrity.

### Index Management

Create snapshots with integrity verification:

```bash
python src/cli.py snapshot --persist-dir data/index --name my_snapshot
```

Verify snapshot integrity:

```bash
python src/cli.py verify-snapshot --name my_snapshot
```

### Local Querying

Query the system via CLI:

```bash
python src/cli.py query "What is malware analysis?"
```

### MCP Server Usage

Start the MCP server:

```bash
python src/mcp/server.py
```

The MCP server will start and listen for connections from MCP clients (IDEs, LLMs).

#### Available Tools

- **query_documents**: Query the RAG system for relevant documents
  - Parameters: `query` (string), `top_k` (integer, 1-10)

#### Available Resources

- **health://status**: Get system health status
- **context://system**: Get system capabilities and context

#### Example MCP Client Usage

```python
from fastmcp import Client

# Connect to MCP server
client = Client("python src/mcp/server.py")

async def query_example():
    async with client:
        # Query documents
        result = await client.call_tool("query_documents", {
            "query": "What is malware analysis?",
            "top_k": 3
        })
        print(result)

        # Get health status
        health = await client.get_resource("health://status")
        print(health)
```

## Configuration

Edit `config/settings.yaml`:

```yaml
data_dir: "./data"
index_dir: "./data/index"
embedding_model: "all-MiniLM-L6-v2"
chunk_size: 1024
vector_store: "faiss"

# MCP server configuration
mcp:
  host: "127.0.0.1"
  port: 25191
  rate_limit:
    max_concurrent: 2       # Maximum concurrent requests
    queue_size: 10          # Maximum pending requests
    default_timeout: 600.0  # Default timeout (seconds)
```

### Key Configuration Sections

- **Data paths**: Configure locations for raw data, processed data, and indexes
- **Embedding**: Model selection, device (CPU/GPU), batch size, caching
- **Chunking**: Adaptive chunking for different content types (code, tables, text)
- **Retrieval**: Hybrid search (semantic + BM25), reranking, filtering
- **MCP**: Server settings, rate limiting, timeouts, tool defaults
- **OCR**: PaddleOCR configuration for image text extraction
- **Boilerplate**: Aggressive removal of web artifacts and document noise

## Testing

Run tests:

```bash
pytest tests/
```

## Requirements

- Python 3.13+

## Docker Deployment

VX-RAG supports deployment via Docker for various usage scenarios.

### Quick Start

```bash
# Build production image
docker build -t vx-rag .

# Run MCP server (STDIO mode - default)
docker run -it \
   -v $(pwd)/data:/app/data \
   -v $(pwd)/logs:/app/logs \
   vx-rag

# Run in HTTP mode for testing
docker compose --profile http up
```

### Available Modes

- **STDIO** (default): IDE integration via MCP client
- **HTTP**: REST API for testing (port 8000)
- **SSE**: Server-Sent Events for web clients

### Ports

- **25191**: Standard MCP server port
- **8000**: HTTP/SSE modes for development

More details: [docs/DOCKER_DEPLOYMENT.md](docs/DOCKER_DEPLOYMENT.md)

## License

The **source code** for this project is licensed under the [MIT](./LICENSE) license.

**Note:** The *dataset* consisting of the raw text files (`.md`, `.pdf`, `.txt`) obtained from [vx-underground.org](https://vx-underground.org/) is distributed under **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.
These materials are provided strictly for **educational and research purposes**, and **commercial use is prohibited**.

See the [DATA LICENSE](./DATA_LICENSE) file for details.
[Official License Text](https://creativecommons.org/licenses/by-nc/4.0/legalcode)
