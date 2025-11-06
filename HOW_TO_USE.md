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

- `query_documents`: search tool
- `health://status`: health check
- `context://system`: system information

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

### Rate Limiting

The current implementation does not include rate limiting or request queuing. For single-user local deployment, this is not a critical issue. Implementation of a queue mechanism for concurrent requests is planned for future releases.

### Request Timeouts

To prevent client hanging, requests are automatically timed out after 10 minutes. This is a safeguard measure - normal queries should complete much faster.

For more information, see `README.md` and the configuration in `config/settings.yaml`.
