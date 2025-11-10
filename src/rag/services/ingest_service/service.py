"""
Ingest Service implementation.

Provides classes and functions for document ingestion.

Supported adapters (per technical standard):
- PDFIngestAdapter: Parses PDF documents using Docling with OCR support
- TXTIngestAdapter: Loads plain text documents  
- MDIngestAdapter: Loads Markdown documents

The system focuses exclusively on local file formats as specified in the
VX-RAG technical standard. API and Database adapters are explicitly excluded
as they are not required for the Vx Underground document collection.
"""

from typing import List, Dict, Any, TYPE_CHECKING, Optional
import logging
import os
from pathlib import Path
import re

if TYPE_CHECKING:
    pass

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption

from ..duplicate_detection_service import DuplicateDetector
from ...libs.utils.text_utils import normalize_text, detect_language
from ..boilerplate_removal_service import remove_boilerplate
from src.utils.config_loader import (
    get_paddle_ocr_config,
    get_boilerplate_removal_config
)

# Lazy import PaddleOCR service
try:
    from ..paddle_ocr_service import PaddleOCRService
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

class IngestAdapter:
    """Base class for ingest adapters."""
    
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        """Load data from source and return unified format."""
        raise NotImplementedError


class PDFIngestAdapter(IngestAdapter):
    """Adapter for loading PDF documents using Docling for local parsing."""
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize PDF adapter with Docling and PaddleOCR.
        Enables OCR for scanned documents as required by the standard.
        
        Args:
            config_path: Path to settings.yaml file
        """
        # Enable OCR for scanned documents (required by standard)
        os.environ['DOCLING_DO_OCR'] = 'true'
        
        # Load PaddleOCR configuration
        self.ocr_config = get_paddle_ocr_config(config_path)
        self.ocr_enabled = self.ocr_config['enabled'] and PADDLE_OCR_AVAILABLE
        
        # Load boilerplate removal configuration
        self.boilerplate_config = get_boilerplate_removal_config(config_path)
        self.boilerplate_enabled = self.boilerplate_config['enabled']
        
        # Initialize PaddleOCR service if enabled
        self.ocr_service: Optional[PaddleOCRService] = None
        if self.ocr_enabled:
            try:
                self.ocr_service = PaddleOCRService(
                    lang=self.ocr_config['lang'],
                    use_gpu=self.ocr_config['use_gpu'],
                    use_angle_cls=self.ocr_config['use_angle_cls'],
                    show_log=self.ocr_config['show_log'],
                    det_model_dir=self.ocr_config['det_model_dir'],
                    rec_model_dir=self.ocr_config['rec_model_dir'],
                    cls_model_dir=self.ocr_config['cls_model_dir'],
                    use_space_char=self.ocr_config['use_space_char'],
                    enable_mkldnn=self.ocr_config['enable_mkldnn'],
                    cpu_threads=self.ocr_config['cpu_threads'],
                    min_confidence=self.ocr_config['min_confidence']
                )
                logger.info("PaddleOCR service initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}. Image text extraction disabled.")
                self.ocr_enabled = False
        else:
            if not PADDLE_OCR_AVAILABLE:
                logger.info("PaddleOCR not available. Install with: pip install paddleocr paddlepaddle")
            else:
                logger.info("PaddleOCR disabled in configuration")
        
        # Configure Docling to extract images
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True  # Enable image extraction
        pipeline_options.images_scale = 2.0  # Higher resolution for better OCR
        
        # Initialize Docling DocumentConverter with image extraction enabled
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
    
    def _process_images_with_ocr(
        self,
        markdown_text: str,
        document: Any,
        pdf_file: Path
    ) -> str:
        """
        Process images from Docling document with PaddleOCR.
        
        Extracts text from images and replaces <!-- image --> placeholders
        with extracted text or code.
        
        Args:
            markdown_text: Original markdown text from Docling
            document: Docling document object with pictures
            pdf_file: Path to source PDF file
            
        Returns:
            Markdown text with image placeholders replaced by extracted text
        """
        if not hasattr(document, 'pictures') or not document.pictures:
            logger.info("No pictures found in document")
            return markdown_text
        
        # Type guard for OCR service
        if self.ocr_service is None:
            logger.warning("OCR service not initialized, skipping image processing")
            return markdown_text
        
        logger.info(f"Processing {len(document.pictures)} images with PaddleOCR")
        
        # Create directory for extracted images if needed
        pdf_images_dir: Optional[Path] = None
        if self.ocr_config['save_extracted_images']:
            images_dir = Path(str(self.ocr_config['extracted_images_dir']))
            pdf_images_dir = images_dir / pdf_file.stem
            pdf_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract text from each image
        image_texts: List[Optional[str]] = []
        for idx, picture in enumerate(document.pictures):
            try:
                # Get image from picture object
                if hasattr(picture, 'image') and hasattr(picture.image, 'pil_image'):
                    pil_image = picture.image.pil_image
                    
                    # Save image if enabled
                    if self.ocr_config['save_extracted_images'] and pdf_images_dir:
                        image_path = pdf_images_dir / f"image_{idx}.png"
                        pil_image.save(str(image_path))
                        logger.debug(f"Saved image to {image_path}")
                    
                    # Extract text with OCR
                    extracted_data = self.ocr_service.extract_text_from_image(
                        pil_image,
                        return_confidence=True
                    )
                    
                    # Handle return type properly
                    if isinstance(extracted_data, dict):
                        extracted_text = str(extracted_data.get('text', ''))
                        confidence = float(extracted_data.get('confidence', 0.0))
                    else:
                        extracted_text = str(extracted_data)
                        confidence = 0.0
                    
                    if extracted_text.strip():
                        # Format as code block (same as Docling does for code)
                        # No special markers - just plain code block
                        formatted_text = f"\n```\n{extracted_text}\n```\n"
                        
                        image_texts.append(formatted_text)
                        logger.info(
                            f"Extracted text from image {idx} "
                            f"(confidence: {confidence:.2f})"
                        )
                    else:
                        # No text found - keep placeholder as is
                        logger.info(f"No text extracted from image {idx}, keeping <!-- image --> placeholder")
                        image_texts.append(None)
                        
                else:
                    logger.warning(f"Cannot access image data for picture {idx}")
                    image_texts.append(None)
                    
            except Exception as e:
                logger.error(f"Failed to process image {idx}: {e}")
                image_texts.append(None)
        
        # Replace image placeholders with extracted text
        if self.ocr_config['replace_image_placeholders'] and image_texts:
            markdown_text = self._replace_image_placeholders(markdown_text, image_texts)
        
        return markdown_text
    
    def _replace_image_placeholders(self, markdown_text: str, image_texts: List[Optional[str]]) -> str:
        """
        Replace <!-- image --> placeholders with extracted text.
        
        Args:
            markdown_text: Original markdown text
            image_texts: List of extracted text from images (None = keep placeholder)
            
        Returns:
            Markdown text with placeholders replaced (or kept if no text extracted)
        """
        # Find all <!-- image --> markers
        pattern = r'<!--\s*image\s*-->'
        matches = list(re.finditer(pattern, markdown_text, re.IGNORECASE))
        
        if not matches:
            logger.info("No <!-- image --> placeholders found in markdown")
            return markdown_text
        
        logger.info(f"Found {len(matches)} image placeholders to replace")
        
        # Replace placeholders from end to start to preserve positions
        result = markdown_text
        for idx, match in enumerate(reversed(matches)):
            # Get corresponding image text (reverse index)
            text_idx = len(matches) - idx - 1
            if text_idx < len(image_texts):
                replacement = image_texts[text_idx]
                # Only replace if text was extracted (not None)
                if replacement is not None:
                    result = result[:match.start()] + replacement + result[match.end():]
                    logger.debug(f"Replaced placeholder at position {match.start()} with extracted text")
                else:
                    logger.debug(f"Keeping <!-- image --> placeholder at position {match.start()} (no text extracted)")
        
        return result
    
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        """
        Load PDF documents from the specified directory using Docling.
        
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
            
            result: List[Dict[str, Any]] = []
            for pdf_file in pdf_files:
                try:
                    logger.info(f"Parsing PDF with Docling: {pdf_file}")
                    
                    # Use Docling to parse the PDF
                    conversion_result = self.converter.convert(str(pdf_file))
                    
                    # Export to markdown
                    markdown_text = conversion_result.document.export_to_markdown()
                    
                    # Process images with PaddleOCR if enabled
                    if self.ocr_enabled and self.ocr_service:
                        markdown_text = self._process_images_with_ocr(
                            markdown_text,
                            conversion_result.document,
                            pdf_file
                        )
                    
                    # Remove boilerplate AFTER OCR but BEFORE normalization
                    # This ensures OCR patterns (<!-- image -->) are preserved during OCR processing
                    if self.boilerplate_enabled:
                        aggressive_mode = self.boilerplate_config['aggressive_mode']
                        cleaned_text = remove_boilerplate(markdown_text, aggressive_mode=aggressive_mode)
                        logger.debug(f"Boilerplate removal applied (aggressive={aggressive_mode})")
                    else:
                        cleaned_text = markdown_text
                        logger.debug("Boilerplate removal disabled")
                    
                    normalized_text = normalize_text(cleaned_text)
                    lang = detect_language(normalized_text)
                    
                    # Extract basic metadata from file
                    title = pdf_file.stem
                    author = 'Unknown'
                    creation_date = None
                    
                    # Get page count from Docling document
                    page_count = len(conversion_result.document.pages) if hasattr(conversion_result.document, 'pages') else None
                    
                    # Count images processed
                    images_processed = len(conversion_result.document.pictures) if hasattr(conversion_result.document, 'pictures') else 0
                    
                    result.append({
                        'id': f"{pdf_file.name}_0",
                        'source': str(pdf_file),
                        'text': normalized_text,
                        'lang': lang,
                        'metadata': {
                            'title': title,
                            'author': author,
                            'creation_date': creation_date,
                            'file_type': 'pdf',
                            'file_name': pdf_file.name,
                            'page_count': page_count,
                            'category': 'document',
                            'parsed_with': 'docling',
                            'content_type': 'markdown',
                            'ocr_enabled': self.ocr_enabled,
                            'images_processed': images_processed
                        }
                    })
                    
                    logger.info(f"Successfully parsed PDF: {pdf_file} (processed {images_processed} images)")
                    
                except Exception as e:
                    logger.error(f"Failed to parse PDF {pdf_file} with Docling: {e}")
            
            logger.info(f"Successfully loaded {len(result)} PDF documents from {raw_pdf_dir} using Docling")
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
                            'file_size': file_path.stat().st_size,
                            'title': file_path.stem,  # Use filename without extension as title
                            'author': 'Unknown',
                            'creation_date': None,
                            'file_type': 'txt',
                            'category': 'text'
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
                            'file_type': 'markdown',
                            'title': file_path.stem,
                            'author': 'Unknown',
                            'creation_date': None,
                            'category': 'markdown'
                        }
                    })
                    
                    logger.info(f"Successfully loaded MD file: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to load MD file {file_path}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load MD documents from {raw_md_dir}: {e}")
            return []


def process_and_save_documents(documents: List[Dict[str, Any]], processed_dir: Path) -> int:
    """
    Process documents (remove duplicates) and save to processed directory.

    Args:
        documents: List of document dictionaries
        processed_dir: Directory to save processed text files

    Returns:
        Number of successfully saved files
    """
    # Remove duplicates
    detector = DuplicateDetector()
    unique_documents = detector.remove_duplicates(documents)

    logger.info(f"After duplicate removal: {len(unique_documents)} unique documents from {len(documents)} total")

    # Save processed documents
    return save_processed_text(unique_documents, processed_dir)


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
