"""
IngestionPipelineService facade.

Provides a simple wrapper around the ingest adapters and exposes a small
API used by the CLI and tasking code. This implementation is intentionally
lightweight to keep unit tests fast and to avoid replicating large LlamaIndex
ingestion pipeline semantics here.

The class exposes the following methods used by the CLI and background
ingest task:
 - process_pdf_directory(Path) -> list[TextNode]
 - process_text_directory(Path, pattern) -> list[TextNode]
 - get_embedder() -> Optional[HuggingFaceEmbedding]
 - persist_pipeline() -> None

This facade wraps the existing adapters in rag.services.ingest_service,
re-using document parsing and normalization logic that already exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import asyncio
import logging

from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.utils.config_loader import get_embedding_config

# Reuse adapters defined in the existing ingest_service
from src.rag.services.ingest_service.service import (
    PDFIngestAdapter,
    TXTIngestAdapter,
    MDIngestAdapter,
)

logger = logging.getLogger(__name__)


class IngestionPipelineService:
    """
    Facade for ingestion pipeline functionality.

    This small facade intentionally keeps behavior minimal and delegates heavy
    lifting to the adapters in `ingest_service.service`.
    """

    def __init__(self, config_path: Optional[str] = None, persist_dir: Optional[str] = None) -> None:
        self.config_path = config_path
        self.persist_dir = persist_dir
        # Initialize adapters lazily
        self._pdf_adapter: Optional[PDFIngestAdapter] = None
        self._txt_adapter: Optional[TXTIngestAdapter] = None
        self._md_adapter: Optional[MDIngestAdapter] = None

    def _get_pdf_adapter(self) -> PDFIngestAdapter:
        if self._pdf_adapter is None:
            self._pdf_adapter = PDFIngestAdapter(config_path=self.config_path)
        return self._pdf_adapter

    def _get_txt_adapter(self) -> TXTIngestAdapter:
        if self._txt_adapter is None:
            self._txt_adapter = TXTIngestAdapter(config_path=self.config_path)
        return self._txt_adapter

    def _get_md_adapter(self) -> MDIngestAdapter:
        if self._md_adapter is None:
            self._md_adapter = MDIngestAdapter(config_path=self.config_path)
        return self._md_adapter

    async def process_pdf_directory(self, directory: Path) -> List[TextNode]:
        """
        Parse PDFs in the given directory and return a list of LlamaIndex
        TextNode objects for downstream processing.
        """
        if not directory.exists() or not directory.is_dir():
            return []

        adapter = self._get_pdf_adapter()
        # `load_data` is synchronous; we call it in a thread to avoid blocking
        raw_docs = await asyncio.to_thread(adapter.load_data, str(directory))

        nodes: List[TextNode] = []
        for d in raw_docs:
            # Expect `d` to be a dict with 'id', 'text', 'metadata'
            text = d.get('text', '')
            metadata = d.get('metadata', {})
            node_id = d.get('id') or d.get('source') or None
            node = TextNode(text=text, metadata=metadata, id_=node_id)
            nodes.append(node)

        return nodes

    async def process_text_directory(self, directory: Path, pattern: str = "*.txt") -> List[TextNode]:
        """
        Parse text/markdown files via adapters and return TextNode list.
        Pattern is used to decide which adapter to invoke.
        """
        if not directory.exists() or not directory.is_dir():
            return []

        pattern = str(pattern).lower()
        if pattern.endswith(".txt") or "txt" in pattern:
            adapter = self._get_txt_adapter()
            raw_docs = await asyncio.to_thread(adapter.load_data, str(directory))
        else:
            adapter = self._get_md_adapter()
            raw_docs = await asyncio.to_thread(adapter.load_data, str(directory))

        nodes: List[TextNode] = []
        for d in raw_docs:
            text = d.get('text', '')
            metadata = d.get('metadata', {})
            node_id = d.get('id') or d.get('source') or None
            node = TextNode(text=text, metadata=metadata, id_=node_id)
            nodes.append(node)

        return nodes

    def get_embedder(self) -> Optional[HuggingFaceEmbedding]:
        """
        Return a configured HuggingFace embedding model if configured.
        Returns None if not configured.
        """
        try:
            embed_cfg = get_embedding_config(self.config_path)
            if not embed_cfg:
                return None
            return HuggingFaceEmbedding(
                model_name=embed_cfg.get('embedding_model', 'all-MiniLM-L6-v2'),
                embed_batch_size=embed_cfg.get('embedding_batch_size', 16),
                trust_remote_code=embed_cfg.get('embedding_trust_remote_code', False),
            )
        except Exception as e:
            logger.warning(f"Failed to initialize embedder: {e}")
            return None

    def persist_pipeline(self) -> None:
        """Persist pipeline state if required (minimal implementation)."""
        # For now we write a minimal pipeline state file to the persist_dir
        if not self.persist_dir:
            return
        try:
            from pathlib import Path
            pdir = Path(self.persist_dir)
            pdir.mkdir(parents=True, exist_ok=True)
            state_file = pdir / "pipeline_state.json"
            import json
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'persisted_at': asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0,
                    'config_path': self.config_path
                }, f)
        except Exception:
            # Nothing to do if we fail to persist
            pass


__all__ = ["IngestionPipelineService"]
