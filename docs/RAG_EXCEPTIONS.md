# RAG Custom Exceptions

This document describes custom exceptions used in the RAG system for more precise error handling.

## Exception Hierarchy

```text
Exception
└── RAGException (base for all RAG errors)
    ├── ServiceInitializationError
    ├── IndexNotFoundError
    ├── IndexLoadError
    ├── RetrievalError
    ├── RerankingError
    ├── EmbeddingError
    ├── ContextAssemblyError
    ├── ConfigurationError
    ├── DocumentParsingError
    ├── VectorStoreError
    └── QueryValidationError
```

## Usage Examples

### Catching Specific Exceptions

```python
from src.rag.orchestrator import get_orchestrator
from src.rag.exceptions import (
    ServiceInitializationError,
    RetrievalError,
    IndexNotFoundError
)

try:
    orchestrator = get_orchestrator()
    results = orchestrator.query("malware analysis")
except IndexNotFoundError as e:
    print(f"Index missing: {e.index_type} at {e.index_path}")
    # Handle missing index (e.g., build new index)
except RetrievalError as e:
    print(f"Query failed: {e.query}")
    print(f"Reason: {e.reason}")
    # Handle retrieval failure (e.g., retry with different parameters)
except ServiceInitializationError as e:
    print(f"Service '{e.service_name}' failed to initialize: {e.reason}")
    # Handle initialization failure (e.g., check dependencies)
```

### Exception Chaining

All custom exceptions support exception chaining with `from`:

```python
try:
    # Some operation
    pass
except ValueError as e:
    raise RetrievalError(query, f"Invalid parameter: {e}") from e
```

This preserves the original exception traceback while providing domain-specific context.

## Exception Attributes

### ServiceInitializationError

- `service_name`: Name of the service that failed
- `reason`: Detailed reason for failure

### IndexNotFoundError

- `index_type`: Type of index (FAISS, BM25, etc.)
- `index_path`: Path where index was expected

### RetrievalError

- `query`: The query string that failed
- `reason`: Detailed reason for failure

### RerankingError

- `reason`: Detailed reason for failure
- `document_count`: Number of documents involved

### EmbeddingError

- `text_preview`: First 100 characters of text that failed
- `reason`: Detailed reason for failure

## Best Practices

1. **Catch Specific Exceptions First**: Always catch more specific exceptions before general ones
2. **Preserve Context**: Use exception chaining (`from e`) to preserve original traceback
3. **Log Before Raising**: Log error details before raising custom exceptions
4. **Provide Actionable Information**: Include enough context to debug and fix issues
5. **Don't Catch RAGException**: Catch specific exception types, not the base class

## Integration with Logging

Custom exceptions work seamlessly with structured logging:

```python
try:
    results = orchestrator.query(query)
except RetrievalError as e:
    logger.error(
        "Query failed",
        query=e.query,
        reason=e.reason,
        exc_info=True
    )
    raise
```
