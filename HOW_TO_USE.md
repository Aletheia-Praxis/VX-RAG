# How to Use VX-RAG

This guide explains how to use the VX-RAG system for processing and searching within a collection of technical documents.

## Prerequisites

- Python 3.13+
- Dependencies installed: `pip install -r requirements.txt`
- Documents in PDF, TXT, or Markdown formats

## Running with Docker

For containerized deployment, use the provided Dockerfile. To ensure data persistence, mount the `data` directory as a volume from your host machine.

1. Build the Docker image:

    ```bash
    docker build -t vx-rag .
    ```

2. Create the data directories on your host:

    ```bash
    mkdir -p data/raw/pdf data/raw/txt data/raw/md data/processed data/index
    ```

3. Run the container with volume mounting:

    ```bash
    docker run -v $(pwd)/data:/app/data -p 5000:5000 vx-rag
    ```

   - `-v $(pwd)/data:/app/data`: Mounts the local `data` directory to `/app/data` in the container for persistence
   - `-p 5000:5000`: Exposes the MCP API port

Note: The container runs as a non-root user for security. Ensure the mounted volume has appropriate permissions if needed.

## Step 1: Prepare Your Documents

1. Create directories for raw documents:

    ```bash
    data/raw/pdf/     # PDF files
    data/raw/txt/     # Text files
    data/raw/md/      # Markdown files
    ```

2. Place your documents in the appropriate directories.

## Step 2: Ingest Documents

Ingestion converts raw documents into processed text.

```bash
python src/cli.py ingest --data-dir data/raw
```

This command:

- Reads PDF, TXT, and MD files
- Extracts text and metadata
- Saves processed files to `data/processed/`

Expected result: "Successfully saved X processed documents to data/processed/"

## Step 3: Build the Index

Create a vector index for fast search.

```bash
python src/cli.py index --persist-dir data/index --data-dir data/processed
```

This command:

- Loads processed documents
- Creates embeddings using sentence-transformers
- Saves the FAISS index

Expected result: "Index created and saved to data/index"

## Step 4: Search and Generate Answers

Run a query against the system.

```bash
python src/cli.py query "What is malware analysis?"
```

This command:

- Searches for relevant documents
- Shows top results with scores

Expected result:

```text
Query: What is malware analysis?

Top Results:
1. Score: 0.95
    Text: Malware analysis involves...
```

## Step 5: MCP Integration (for Developers)

Start the MCP server for IDE or LLM integration.

```bash
python src/mcp/server.py
```

The server provides:

- `query_knowledge_base`: Advanced semantic search with reranking
- `search_documents`: Fast document search
- Health and status endpoints

### MCP Configuration

The MCP server can be configured in `config/settings.yaml`:

```yaml
mcp:
  host: "127.0.0.1"
  port: 25191
  rate_limit:
    max_concurrent: 2       # Adjust for your workload
    queue_size: 10          # Queue capacity
    default_timeout: 600.0  # 10 minutes
```

The middleware automatically handles:

- Request rate limiting and queuing
- Structured logging with request IDs
- Performance metrics collection
- Timeout management

## Troubleshooting

### Index Not Loading

- Ensure the index is created using the `index` command
- Check for files in `data/index/`

### No Search Results

- Verify documents are processed (`data/processed/`)
- Try a simpler query

### Dependency Errors

- Check Python and package versions

## Additional Features

- **Filters**: Add metadata to filter by author, date, etc.
- **Reranking**: Automatic improvement of search results
- **Batch Processing**: For large document collections

## Known Limitations and Future Improvements

### Rate Limiting and Request Management

The system implements rate limiting with request queuing to prevent resource exhaustion:

- **Maximum Concurrent Requests**: 2 (configurable in `config/settings.yaml`)
- **Queue Size**: 10 pending requests
- **Default Timeout**: 10 minutes (600 seconds)
- **Tool-Specific Timeouts**:
  - Knowledge base queries: 10 minutes
  - Document search: 5 minutes
  - Health checks: 30 seconds

When the system is at capacity, new requests are queued automatically. If the queue is full, requests are rejected with an error. This ensures stable operation even under load.

For multi-user deployment, these limits can be adjusted in the `mcp.rate_limit` section of `config/settings.yaml`.

### Request Timeouts

Requests are automatically timed out based on operation type to prevent resource hanging. Normal queries typically complete in seconds, but complex operations (e.g., large document searches) may take longer. Timeout values can be customized per tool in the `mcp.timeouts` configuration section.

For more information, see `README.md` and the configuration in `config/settings.yaml`.
