"""
Ingest Service implementation.

Provides classes and functions for document ingestion.
"""

from typing import List, Dict, Any, TYPE_CHECKING
import logging
from pathlib import Path

if TYPE_CHECKING:
    import pandas as pd

from llama_index.core import SimpleDirectoryReader

from ...libs.utils.text_utils import normalize_text, detect_language

logger = logging.getLogger(__name__)

class IngestAdapter:
    """Base class for ingest adapters."""
    
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        """Load data from source and return unified format."""
        raise NotImplementedError


class PDFIngestAdapter(IngestAdapter):
    """Adapter for loading PDF documents."""
    
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        """
        Load PDF documents from the specified directory.
        
        Args:
            source: Path to the directory containing PDF files
            
        Returns:
            List of document dictionaries with text and metadata
        """
        raw_pdf_dir = Path(source)
        
        if not raw_pdf_dir.exists():
            logger.error(f"Raw PDF directory does not exist: {raw_pdf_dir}")
            return []

        if not raw_pdf_dir.is_dir():
            logger.error(f"Raw PDF path is not a directory: {raw_pdf_dir}")
            return []

        try:
            logger.info(f"Scanning directory {raw_pdf_dir} for PDF files")
            # List all files to log ignored ones
            all_files = list(raw_pdf_dir.glob("*"))
            pdf_files = [f for f in all_files if f.suffix.lower() == '.pdf']
            other_files = [f for f in all_files if f.suffix.lower() != '.pdf' and f.is_file()]
            
            if other_files:
                logger.info(f"Found {len(pdf_files)} PDF files and {len(other_files)} other files (ignored): {[f.name for f in other_files]}")
            else:
                logger.info(f"Found {len(pdf_files)} PDF files, no other files to ignore")
            
            reader = SimpleDirectoryReader(
                input_dir=str(raw_pdf_dir),
                required_exts=[".pdf"],
                recursive=False  # Only process files directly in the directory
            )
            documents = reader.load_data()
            logger.info(f"Successfully loaded {len(documents)} PDF documents from {raw_pdf_dir}")
            
            # Convert to unified format
            result = []
            for doc in documents:
                normalized_text = normalize_text(doc.text)
                lang = detect_language(normalized_text)
                result.append({
                    'id': doc.id_,
                    'source': doc.metadata.get('file_path', 'unknown'),
                    'text': normalized_text,
                    'lang': lang,
                    'metadata': doc.metadata
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load PDF documents from {raw_pdf_dir}: {e}")
            return []


class TXTIngestAdapter(IngestAdapter):
    """Adapter for loading TXT documents."""
    
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        """
        Load TXT documents from the specified directory.
        
        Args:
            source: Path to the directory containing TXT files
            
        Returns:
            List of document dictionaries with unified format
        """
        raw_txt_dir = Path(source)
        
        if not raw_txt_dir.exists():
            logger.error(f"Raw TXT directory does not exist: {raw_txt_dir}")
            return []

        if not raw_txt_dir.is_dir():
            logger.error(f"Raw TXT path is not a directory: {raw_txt_dir}")
            return []

        try:
            logger.info(f"Scanning directory {raw_txt_dir} for TXT files")
            txt_files = list(raw_txt_dir.glob("*.txt"))
            logger.info(f"Found {len(txt_files)} TXT files")
            
            result = []
            for file_path in txt_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    normalized_text = normalize_text(text)
                    lang = detect_language(normalized_text)
                    
                    result.append({
                        'id': str(file_path),
                        'source': str(file_path),
                        'text': normalized_text,
                        'lang': lang,
                        'metadata': {
                            'file_path': str(file_path),
                            'file_name': file_path.name,
                            'file_size': file_path.stat().st_size
                        }
                    })
                    
                    logger.info(f"Successfully loaded TXT file: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to load TXT file {file_path}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load TXT documents from {raw_txt_dir}: {e}")
            return []


class MDIngestAdapter(IngestAdapter):
    """Adapter for loading Markdown documents."""
    
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        """
        Load Markdown documents from the specified directory.
        
        Args:
            source: Path to the directory containing MD files
            
        Returns:
            List of document dictionaries with unified format
        """
        raw_md_dir = Path(source)
        
        if not raw_md_dir.exists():
            logger.error(f"Raw MD directory does not exist: {raw_md_dir}")
            return []

        if not raw_md_dir.is_dir():
            logger.error(f"Raw MD path is not a directory: {raw_md_dir}")
            return []

        try:
            logger.info(f"Scanning directory {raw_md_dir} for MD files")
            md_files = list(raw_md_dir.glob("*.md"))
            logger.info(f"Found {len(md_files)} MD files")
            
            result = []
            for file_path in md_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # For Markdown, we can optionally parse headers or keep as is
                    # For now, treat as plain text
                    normalized_text = normalize_text(text)
                    lang = detect_language(normalized_text)
                    
                    result.append({
                        'id': str(file_path),
                        'source': str(file_path),
                        'text': normalized_text,
                        'lang': lang,
                        'metadata': {
                            'file_path': str(file_path),
                            'file_name': file_path.name,
                            'file_size': file_path.stat().st_size,
                            'file_type': 'markdown'
                        }
                    })
                    
                    logger.info(f"Successfully loaded MD file: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to load MD file {file_path}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load MD documents from {raw_md_dir}: {e}")
            return []


class APIIngestAdapter(IngestAdapter):
    """Adapter for loading data from JSON APIs."""
    
    def __init__(self, timeout: int = 30, retries: int = 3):
        """
        Initialize API adapter.
        
        Args:
            timeout: Request timeout in seconds
            retries: Number of retry attempts
        """
        self.timeout = timeout
        self.retries = retries
    
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        """
        Load data from JSON API endpoint.
        
        Args:
            source: API endpoint URL
            
        Returns:
            List of document dictionaries with unified format
        """
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        try:
            # Setup retry strategy
            retry_strategy = Retry(
                total=self.retries,
                status_forcelist=[429, 500, 502, 503, 504],
                backoff_factor=1
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            
            with requests.Session() as session:
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                
                logger.info(f"Fetching data from API: {source}")
                response = session.get(source, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                
                # Assume data is a list of items or a single item
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = [data]
                else:
                    logger.error(f"Unexpected API response format from {source}")
                    return []
                
                result = []
                for i, item in enumerate(items):
                    # Extract text from item (customize based on API structure)
                    text = self._extract_text_from_item(item)
                    if text:
                        normalized_text = normalize_text(text)
                        lang = detect_language(normalized_text)
                        
                        result.append({
                            'id': f"{source}_{i}",
                            'source': source,
                            'text': normalized_text,
                            'lang': lang,
                            'metadata': {
                                'api_url': source,
                                'item_index': i,
                                'raw_data': item
                            }
                        })
                
                logger.info(f"Successfully loaded {len(result)} items from API: {source}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to load data from API {source}: {e}")
            return []
    
    def _extract_text_from_item(self, item: Dict[str, Any]) -> str:
        """
        Extract text content from API item.
        Customize this method based on the API response structure.
        
        Args:
            item: API response item
            
        Returns:
            Extracted text
        """
        # Default implementation: look for common text fields
        text_fields = ['text', 'content', 'description', 'body', 'message']
        for field in text_fields:
            if field in item:
                value = item[field]
                if isinstance(value, str):
                    return value
        
        # If no text field found, convert the whole item to string
        return str(item)


class DatabaseIngestAdapter(IngestAdapter):
    """Adapter for loading data from databases."""
    
    def __init__(self, connection_string: str, query: str):
        """
        Initialize database adapter.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
        """
        self.connection_string = connection_string
        self.query = query
    
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        """
        Load data from database using the configured query.
        
        Args:
            source: Ignored for database adapter (uses configured connection)
            
        Returns:
            List of document dictionaries with unified format
        """
        try:
            import pandas as pd
            
            logger.info(f"Connecting to database and executing query")
            df = pd.read_sql(self.query, self.connection_string)
            
            result = []
            for idx, row in df.iterrows():
                # Convert row to text (customize based on schema)
                text = self._row_to_text(row)
                if text:
                    normalized_text = normalize_text(text)
                    lang = detect_language(normalized_text)
                    
                    result.append({
                        'id': f"db_{idx}",
                        'source': self.connection_string.split('@')[-1] if '@' in self.connection_string else 'database',
                        'text': normalized_text,
                        'lang': lang,
                        'metadata': {
                            'db_connection': self.connection_string,
                            'query': self.query,
                            'row_index': idx,
                            'raw_row': row.to_dict()
                        }
                    })
            
            logger.info(f"Successfully loaded {len(result)} rows from database")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            return []
    
    def _row_to_text(self, row: "pd.Series") -> str:
        """
        Convert database row to text content.
        Customize this method based on your database schema.
        
        Args:
            row: Pandas Series representing a database row
            
        Returns:
            Extracted text
        """
        # Default: concatenate all string columns
        text_parts = []
        for col, value in row.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(f"{col}: {value}")
        
        return ' '.join(text_parts)


def save_processed_text(documents: List[Dict[str, Any]], processed_dir: Path) -> int:
    """
    Save the text content of documents to processed directory as .txt files.
    Also saves metadata as .json files.

    Args:
        documents: List of document dictionaries with unified format
        processed_dir: Directory to save processed text files

    Returns:
        Number of successfully saved files
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    total_docs = len(documents)

    logger.info(f"Starting to save {total_docs} documents to {processed_dir}")

    for i, doc in enumerate(documents, 1):
        try:
            # Use id as filename base
            base_name = Path(doc['id']).name.replace('/', '_').replace('\\', '_')
            txt_path = processed_dir / f"{base_name}.txt"
            json_path = processed_dir / f"{base_name}.json"

            # Check if text is not empty
            text = doc.get('text', '').strip()
            if not text:
                logger.warning(f"Document {doc['id']} has no extractable text, skipping")
                continue

            # Save text
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # Save metadata
            metadata = {
                'id': doc['id'],
                'source': doc['source'],
                'lang': doc['lang'],
                'file_path': txt_path.name,
                'additional_metadata': doc['metadata']
            }
            import json
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved processed text and metadata ({i}/{total_docs}) to {txt_path} and {json_path}")
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to save document {doc['id']}: {e}")

    logger.info(f"Successfully saved {saved_count}/{total_docs} documents")
    return saved_count
