from src.utils.logging_config import get_logger
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document as LlamaDocument
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.datamodel.base_models import InputFormat

from src.rag.services.boilerplate_removal_service.service import BoilerplateRemovalService

logger = get_logger(__name__)

class DoclingReader(BaseReader):
    """
    LlamaIndex-compatible reader for PDF files using Docling.
    """
    def __init__(
        self, 
        ocr_enabled: bool = True,
        boilerplate_service: Optional[BoilerplateRemovalService] = None
    ):
        self.ocr_enabled = ocr_enabled
        self.boilerplate_service = boilerplate_service
        
        # Configure Docling options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ocr_enabled
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def load_data(self, file: Path, extra_info: Optional[Dict] = None) -> List[LlamaDocument]:
        """
        Load data from a PDF file.
        """
        try:
            logger.info(f"Processing PDF with Docling: {file}")
            
            # Convert using Docling
            conversion_result = self.converter.convert(file)
            
            # Export to Markdown
            markdown_text = conversion_result.document.export_to_markdown()
            
            # Apply boilerplate removal if enabled
            if self.boilerplate_service:
                markdown_text = self.boilerplate_service.remove_boilerplate(markdown_text)
                
            # Create LlamaIndex Document
            metadata = {
                "source": str(file),
                "filename": file.name,
                "extension": file.suffix,
                "parsed_with": "docling",
                "content_type": "markdown" # Docling exports to markdown
            }
            
            if extra_info:
                metadata.update(extra_info)
                
            return [LlamaDocument(text=markdown_text, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            return []

def process_and_save_documents(documents: List[Dict[str, Any]], output_dir: Path) -> int:
    """
    Save processed documents to JSON file.
    
    Args:
        documents: List of document dictionaries
        output_dir: Directory to save the output file
        
    Returns:
        Number of documents saved
    """
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "nodes.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
        
    return len(documents)
