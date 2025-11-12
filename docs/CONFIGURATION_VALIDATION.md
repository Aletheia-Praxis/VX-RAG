# Configuration Validation

## Overview

VX-RAG uses **Pydantic v2** to validate all settings loaded from `config/settings.yaml`. This provides:

- **Strict typing**: All parameters have explicit types
- **Automatic validation**: Invalid values are detected during loading
- **Clear error messages**: Detailed descriptions of validation issues
- **Default values**: Automatic application of sensible defaults
- **Range validation**: Ensures values are within allowed boundaries

## Architecture

### Components

1. **`src/utils/config_schemas.py`**: Pydantic models for validation
2. **`src/utils/config_loader.py`**: Configuration loading and validation
3. **`config/settings.yaml`**: Configuration file

### Flow

```txt
settings.yaml → YAML Parser → Pydantic Validation → Validated Config
                                        ↓
                                 ValidationError (if invalid)
```

## Pydantic Models

### Main Configuration Model

```python
class VXRAGSettings(BaseModel):
    """Complete VX-RAG configuration settings."""
    
    # Data directories
    data_dir: str = Field(default="./data")
    raw_data_dir: str = Field(default="./data/raw")
    
    # Embedding
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_device: Literal["cpu", "cuda"] = Field(default="cpu")
    embedding_batch_size: int = Field(default=10, ge=1, le=1000)
    
    # Chunking
    chunk_size: int = Field(default=1024, ge=128, le=4096)
    chunk_overlap: int = Field(default=200, ge=0)
    
    # Nested configurations
    faiss: FAISSConfig
    mcp: MCPConfig
    retriever: RetrieverConfig
    # ... other subsystems
```

### Subsystem Models

Each subsystem has its own Pydantic model with specific validation:

- **EmbeddingConfig**: Embedding model settings
- **ChunkingConfig**: Document chunking parameters
- **FAISSConfig**: Vector store configuration
- **RetrieverConfig**: Search and retrieval settings
- **RerankerConfig**: Reranking configuration
- **MCPConfig**: MCP server settings
- **PaddleOCRConfig**: OCR settings
- **BoilerplateRemovalConfig**: Content cleaning settings

## Validation Rules

### Numeric Ranges

```python
# Embedding batch size: 1-1000
embedding_batch_size: int = Field(default=10, ge=1, le=1000)

# FAISS HNSW parameter: 4-128
hnsw_m: int = Field(default=32, ge=4, le=128)

# Hybrid search alpha: 0.0-1.0
hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

# MCP server port: 1024-65535
port: int = Field(default=25191, ge=1024, le=65535)
```

### Literal Types

```python
# Device must be "cpu" or "cuda"
embedding_device: Literal["cpu", "cuda"] = Field(default="cpu")

# Metric must be specific value
metric: Literal["inner_product", "L2"] = Field(default="inner_product")

# Log level must be valid
log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
```

### Custom Validators

```python
@model_validator(mode='after')
def validate_overlap_size(self) -> 'ChunkingConfig':
    """Ensure overlap is less than chunk size."""
    if self.chunk_overlap >= self.chunk_size:
        raise ValueError(
            f"chunk_overlap ({self.chunk_overlap}) must be less than "
            f"chunk_size ({self.chunk_size})"
        )
    return self
```

## Usage

### Basic Usage

```python
from src.utils.config_loader import load_settings, get_validated_settings

# Load and validate configuration
config_dict = load_settings()  # Returns validated dict

# Get Pydantic model instance
settings = get_validated_settings()  # Returns VXRAGSettings instance

# Access configuration
print(settings.embedding_model)
print(settings.mcp.port)
```

### Handling Validation Errors

```python
from pydantic import ValidationError
from src.utils.config_loader import load_settings

try:
    config = load_settings("path/to/settings.yaml")
except FileNotFoundError as e:
    print(f"Config file not found: {e}")
except ValueError as e:
    print(f"Configuration validation failed: {e}")
```

### Example Validation Error

```txt
Configuration validation errors:
2 validation errors for VXRAGSettings
chunk_overlap
  Value error, chunk_overlap (512) must be less than chunk_size (512)
embedding_batch_size
  Input should be less than or equal to 1000 [type=less_than_equal, input_value=2000]
```

## Configuration Examples

### Minimal Configuration

```yaml
# settings.yaml - Uses defaults for most values
data_dir: "./data"
chunk_size: 1024
```

### Custom Configuration

```yaml
data_dir: "./data"
embedding_model: "all-MiniLM-L6-v2"
embedding_device: "cpu"
embedding_batch_size: 20

chunk_size: 2048
chunk_overlap: 400

faiss:
  hnsw_m: 64
  metric: "inner_product"

mcp:
  host: "127.0.0.1"
  port: 25191
  rate_limit:
    max_concurrent: 5
    queue_size: 20

retriever:
  semantic_top_k: 30
  hybrid_alpha: 0.7
  enable_hybrid: true
```

## Common Validation Errors

### 1. Overlap >= Chunk Size

**Invalid:**

```yaml
chunk_size: 512
chunk_overlap: 512
```

**Valid:**

```yaml
chunk_size: 512
chunk_overlap: 200
```

### 2. Out of Range Values

**Invalid:**

```yaml
embedding_batch_size: 2000  # Max is 1000
```

**Valid:**

```yaml
embedding_batch_size: 100
```

### 3. Invalid Device

**Invalid:**

```yaml
embedding_device: "gpu"  # Must be "cpu" or "cuda"
```

**Valid:**

```yaml
embedding_device: "cuda"
```

### 4. Invalid Port

**Invalid:**

```yaml
mcp:
  port: 80  # Below 1024
```

**Valid:**

```yaml
mcp:
  port: 25191
```

## Testing

### Unit Tests

Tests are located in:

- `tests/unit_test/utils_test/test_config_schemas.py` - Pydantic model tests
- `tests/unit_test/utils_test/test_config_loader_validation.py` - Loader tests

Run tests:

```bash
pytest tests/unit_test/utils_test/test_config_schemas.py -v
pytest tests/unit_test/utils_test/test_config_loader_validation.py -v
```

### Test Coverage

- Valid configurations
- Invalid value ranges
- Invalid literal types
- Custom validator logic
- Nested configuration validation
- Default value application
- Extra fields handling
- Error message formatting

## Benefits

### 1. Early Error Detection

Configuration issues are detected at startup, not during runtime.

### 2. Type Safety

Pydantic provides strict typing, helping IDEs and mypy.

### 3. Self-Documenting

Pydantic models serve as documentation for available parameters.

### 4. Consistent Defaults

Centralized default values in Pydantic schemas.

### 5. Better IDE Support

Autocomplete and type checking in IDEs.

## Best Practices

### 1. Always Use Validated Settings

```python
# Don't bypass validation
with open("settings.yaml") as f:
    config = yaml.safe_load(f)  # No validation!

# Use validated loader
from src.utils.config_loader import load_settings
config = load_settings()  # Validated!
```

### 2. Access via get_validated_settings()

```python
# Get typed settings instance
from src.utils.config_loader import get_validated_settings

settings = get_validated_settings()
print(settings.mcp.port)  # Type-safe access with IDE support
```

### 3. Handle Validation Errors

```python
try:
    settings = get_validated_settings()
except ValueError as e:
    logger.error(f"Invalid configuration: {e}")
    # Provide meaningful error to user
    sys.exit(1)
```

### 4. Use Field Descriptions

```python
# Good: Descriptive field with validation
port: int = Field(
    default=25191,
    ge=1024,
    le=65535,
    description="MCP server port"
)
```

## Extending Configuration

### Adding New Fields

1. Add field to appropriate Pydantic model:

```python
class MCPConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=25191, ge=1024, le=65535)
    # New field
    ssl_enabled: bool = Field(default=False, description="Enable SSL/TLS")
```

2. Add to `settings.yaml`:

```yaml
mcp:
  host: "127.0.0.1"
  port: 25191
  ssl_enabled: true
```

3. Add tests:

```python
def test_ssl_enabled() -> None:
    config = MCPConfig(ssl_enabled=True)
    assert config.ssl_enabled is True
```

### Adding Custom Validators

```python
class CustomConfig(BaseModel):
    min_value: int
    max_value: int
    
    @model_validator(mode='after')
    def validate_range(self) -> 'CustomConfig':
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
        return self
```

## Migration from Dict-Based Config

### Before (Dict-based)

```python
config = yaml.safe_load(open("settings.yaml"))
batch_size = config.get("embedding", {}).get("batch_size", 10)
```

### After (Pydantic)

```python
settings = get_validated_settings()
batch_size = settings.embedding_batch_size  # Type-safe, validated
```

## References

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pydantic Field Validation](https://docs.pydantic.dev/latest/concepts/fields/)
- [Pydantic Model Validators](https://docs.pydantic.dev/latest/concepts/validators/)
