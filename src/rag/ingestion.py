"""
Conditional Docling Ingestion Pipeline & Adaptive Markdown Chunker for VX-RAG.

Authoritative Source: ORIGINAL_REQUEST.md §R2, PROJECT.md Features 6-10, Tech Spec §4.1-§4.3.
Enforces:
- Docling programmatic parsing as primary standard extraction (PDF, TXT, MD).
- Conditional OCR fallback to Tesseract ONLY when text stream is empty or document is
  an image or copy-protected. Searchable text PDFs MUST NOT invoke OCR.
- Boilerplate removal: aggressive stripping of page numbers, recurring headers, footers,
  while strictly preserving markdown headers, tables, and code blocks.
- Markdown-aware chunking: 1024 tokens default with 10-15% overlap, respecting sentence
  boundaries and heading context.
- Technical adaptive chunking: 256-512 tokens for code blocks, preserving code fences
  intact without splitting mid-syntax.
- Strict metadata assignment: every generated chunk TextNode contains ONLY and EXACTLY
  the 5 mandatory schema fields from src.rag.metadata.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from llama_index.core.schema import TextNode

from src.rag.exceptions import DocumentParsingError
from src.rag.metadata import (
    STRICT_METADATA_KEYS,
    compute_file_hash,
    extract_document_metadata,
    sanitize_node_metadata,
)
from src.utils.logging_config import get_logger

logger = get_logger("ingestion")

# Named constants for chunking budgets and overlap ratios
DEFAULT_CHUNK_SIZE: int = 1024
DEFAULT_CODE_CHUNK_SIZE: int = 512
DEFAULT_CHUNK_OVERLAP: int = 128
MIN_CODE_CHUNK_SIZE: int = 256
MAX_CODE_CHUNK_SIZE: int = 512
DEFAULT_OVERLAP_RATIO_MIN: float = 0.10
DEFAULT_OVERLAP_RATIO_MAX: float = 0.15
MAX_UNBROKEN_STRING_LENGTH: int = 1000

# Regex patterns for boilerplate removal
PAGE_NUMBER_PATTERN: re.Pattern[str] = re.compile(
    r"(?m)^\s*(?:Page\s+\d+(?:\s+of\s+\d+)?|-?\s*\d+\s*-?)\s*$"
)
CONFIDENTIAL_BANNER_PATTERN: re.Pattern[str] = re.compile(
    r"(?m)^\s*CONFIDENTIAL\s*-\s*DO NOT DISTRIBUTE\s*$"
)
EXTRA_BANNER_PATTERN: re.Pattern[str] = re.compile(
    r"(?m)^\s*(?:TOP SECRET|RESTRICTED|CLASSIFIED)\s*-\s*DO NOT DISTRIBUTE\s*$"
)
HTML_COMMENT_PATTERN: re.Pattern[str] = re.compile(r"<!--.*?-->", re.DOTALL)
MULTIPLE_NEWLINES_PATTERN: re.Pattern[str] = re.compile(r"\n{3,}")

# Regex patterns for structural markdown identification
CODE_BLOCK_PATTERN: re.Pattern[str] = re.compile(
    r"(?ms)(^```[^\n]*\n.*?^```$)"
)
SENTENCE_SPLIT_PATTERN: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+")
HEX_DUMP_ROW_PATTERN: re.Pattern[str] = re.compile(
    r"^[0-9a-fA-F]{6,8}\s+[0-9a-fA-F\s]{10,}\|.*\|$"
)

# Supported image file extensions that require OCR
IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
TEXT_EXTENSIONS: set[str] = {".txt", ".md", ".markdown"}


class CorruptedDocumentError(DocumentParsingError, ValueError):
    """Exception raised when a document has corrupted header or invalid binary format."""

    def __init__(self, file_path: str, reason: str) -> None:
        """
        Initialize corrupted document error.

        Args:
            file_path: Path to the target document.
            reason: Explanation of the corruption.
        """
        super().__init__(file_path, reason)


def count_tokens(text: str) -> int:
    """
    Count tokens in text conservatively.

    Uses maximum of tiktoken encoding tokens and whitespace-separated words to guarantee
    budget compliance for both subword token models and word-based evaluation criteria.

    Args:
        text: Input text string to count.

    Returns:
        Token count integer.
    """
    if not text:
        return 0
    words = len(text.split())
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        toks = len(enc.encode(text))
        return max(words, toks)
    except (ImportError, RuntimeError, ValueError):
        return words


def balance_code_fences(text: str) -> str:
    """
    Ensure code fences are properly balanced by closing unclosed blocks.

    Prevents syntax runaway and protects chunking AST state.

    Args:
        text: Input markdown text.

    Returns:
        Balanced markdown text.
    """
    fence_count = len(re.findall(r"^```", text, flags=re.MULTILINE))
    if fence_count % 2 != 0:
        text = text.rstrip() + "\n```\n"
    return text


def break_unbroken_strings(text: str, max_chunk_chars: int = MAX_UNBROKEN_STRING_LENGTH) -> str:
    """
    Safely split unbroken strings longer than max_chunk_chars to prevent infinite loops.

    Args:
        text: Input string.
        max_chunk_chars: Maximum continuous character length before safe division.

    Returns:
        Text with long unbroken tokens segmented by spaces.
    """
    if not text or len(text) <= max_chunk_chars:
        return text

    words = text.split(" ")
    safe_words: list[str] = []
    for word in words:
        if len(word) > max_chunk_chars:
            for i in range(0, len(word), max_chunk_chars):
                safe_words.append(word[i : i + max_chunk_chars])
        else:
            safe_words.append(word)
    return " ".join(safe_words)


class DoclingPipeline:
    """
    Conditional Docling Ingestion Pipeline with adaptive Markdown chunking.

    Processes PDF, TXT, MD documents using standard programmatic extraction first,
    falling back to Tesseract OCR strictly when standard extraction yields no text
    or document is copy-protected/image-only.
    """

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        default_chunk_size: int = DEFAULT_CHUNK_SIZE,
        code_chunk_size: int = DEFAULT_CODE_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """
        Initialize Docling ingestion pipeline.

        Args:
            config_path: Path to configuration YAML file.
            default_chunk_size: Target token size for general text chunks.
            code_chunk_size: Target token window for code chunks (256-512).
            overlap: Number of tokens to overlap between adjacent chunks.

        Raises:
            ValueError: If overlap is greater than or equal to chunk size, or negative.
        """
        if overlap >= default_chunk_size:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be strictly less than chunk_size ({default_chunk_size})"
            )
        if overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if default_chunk_size <= 0:
            raise ValueError("default_chunk_size must be strictly positive")
        if code_chunk_size <= 0:
            raise ValueError("code_chunk_size must be strictly positive")

        self.config_path = config_path
        self.default_chunk_size = default_chunk_size
        self.code_chunk_size = code_chunk_size
        self.overlap = overlap

        self._ocr_invoked: bool = False
        self._docling_converter: Any | None = None
        self._docling_ocr_converter: Any | None = None

    @property
    def ocr_invoked(self) -> bool:
        """
        Check if OCR was triggered during the last extraction.

        Returns:
            True if OCR fallback was invoked, False otherwise.
        """
        return self._ocr_invoked

    def _get_docling_converter(self, enable_ocr: bool = False) -> Any | None:
        """
        Get or initialize Docling DocumentConverter instance.

        Args:
            enable_ocr: Whether to configure OCR in pipeline options.

        Returns:
            DocumentConverter instance or None if docling is unavailable.
        """
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TesseractCliOcrOptions,
                TesseractOcrOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            if not enable_ocr and self._docling_converter is not None:
                return self._docling_converter
            if enable_ocr and self._docling_ocr_converter is not None:
                return self._docling_ocr_converter

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = enable_ocr
            pipeline_options.do_table_structure = True

            if enable_ocr:
                try:
                    pipeline_options.ocr_options = TesseractCliOcrOptions()
                except (ImportError, AttributeError, ValueError, RuntimeError) as cli_err:
                    logger.debug(f"Tesseract CLI OCR options unavailable: {cli_err}")
                    try:
                        pipeline_options.ocr_options = TesseractOcrOptions()
                    except (ImportError, AttributeError, ValueError, RuntimeError) as opt_err:
                        logger.debug(f"Tesseract OCR options fallback: {opt_err}")

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            if enable_ocr:
                self._docling_ocr_converter = converter
            else:
                self._docling_converter = converter

            return converter
        except ImportError:
            logger.debug("Docling library not installed, using fallback extraction")
            return None
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning(f"Failed to initialize Docling converter: {exc}")
            return None

    def _ocr_extract(self, file_path: Path) -> str:
        """
        Perform OCR extraction using Tesseract engine via Docling.

        Args:
            file_path: Path to the target document.

        Returns:
            Extracted text string from OCR.
        """
        converter = self._get_docling_converter(enable_ocr=True)
        if converter is not None:
            res = converter.convert(str(file_path))
            return str(res.document.export_to_markdown())
        return ""

    def _perform_ocr_fallback(self, file_path: Path) -> str:
        """
        Execute conditional OCR fallback and format output into clean Markdown.

        Gracefully catches any OCR engine failures, logging a warning without crashing.

        Args:
            file_path: Path to the document requiring OCR.

        Returns:
            Extracted text formatted as clean Markdown, or empty string on failure.
        """
        self._ocr_invoked = True
        try:
            raw_ocr = self._ocr_extract(file_path)
            if not raw_ocr or not raw_ocr.strip():
                return ""
            clean_ocr = raw_ocr.strip()
            if not clean_ocr.startswith("#"):
                return f"## OCR Ingested Content\n\n{clean_ocr}"
            return clean_ocr
        except (RuntimeError, ValueError, OSError, TypeError) as exc:
            logger.warning(f"Conditional OCR extraction failed for {file_path}: {exc}")
            return ""

    def extract_text(self, file_path: Path | str) -> str:
        """
        Extract text from document using programmatic extraction first.

        Only falls back to Tesseract OCR if text stream is empty or document is
        copy-protected/image-only. Searchable text PDFs will NOT trigger OCR.

        Args:
            file_path: Path to the document.

        Returns:
            Extracted text content string.

        Raises:
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If the target path is a directory.
            CorruptedDocumentError: If document binary header is corrupted.
            DocumentParsingError: If document cannot be parsed.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"Target path is a directory: {path}")

        suffix = path.suffix.lower()

        # Handle plaintext and markdown files (standard programmatic extraction)
        if suffix in TEXT_EXTENSIONS:
            self._ocr_invoked = False
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return path.read_text(encoding="latin-1")

        # Handle images: no programmatic text layer exists, directly trigger OCR fallback
        if suffix in IMAGE_EXTENSIONS:
            return self._perform_ocr_fallback(path)

        # Handle PDF documents
        if suffix == ".pdf":
            raw_bytes = path.read_bytes()
            if len(raw_bytes) == 0:
                raise DocumentParsingError(str(path), f"Empty 0-byte PDF document: {path.name}")
            if not raw_bytes.startswith(b"%PDF"):
                raise CorruptedDocumentError(str(path), f"Corrupted PDF header in {path.name}")

            # Inspect PDF text layer and permissions via pypdf
            is_copy_protected = False
            has_text_layer = False
            programmatic_text = ""

            try:
                import pypdf

                reader = pypdf.PdfReader(str(path))
                if reader.is_encrypted:
                    try:
                        decrypt_code = reader.decrypt("")
                        if decrypt_code == 0:
                            is_copy_protected = True
                    except (RuntimeError, ValueError, OSError, TypeError):
                        is_copy_protected = True

                if not is_copy_protected:
                    page_texts: list[str] = []
                    for page in reader.pages:
                        page_str = page.extract_text() or ""
                        if page_str.strip():
                            page_texts.append(page_str.strip())
                    if page_texts:
                        programmatic_text = "\n\n".join(page_texts).strip()
                        has_text_layer = len(programmatic_text) > 0
            except (RuntimeError, ValueError, OSError, TypeError) as exc:
                logger.debug(f"PDF programmatic extraction check exception: {exc}")

            # Conditional check: Searchable text PDFs MUST NOT invoke OCR!
            if has_text_layer and not is_copy_protected:
                self._ocr_invoked = False
                return programmatic_text

            # Trigger conditional OCR fallback for image-only or copy-protected PDFs
            return self._perform_ocr_fallback(path)

        # Fallback for unknown file extensions
        try:
            self._ocr_invoked = False
            return path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError, ValueError):
            return self._perform_ocr_fallback(path)

    def clean_boilerplate(self, text: str) -> str:
        """
        Strip headers, footers, and page numbers while preserving markdown and code.

        Strictly preserves:
        - Markdown headers (#, ##, ###, etc.)
        - Fenced code blocks (```...```) and formatting
        - Markdown tables (| Col | ... |)
        - Hex memory dumps (00000000  4d 5a ... |...|)

        Args:
            text: Raw input text.

        Returns:
            Cleaned text with boilerplate removed.
        """
        if not text:
            return ""

        # Protect fenced code blocks by isolating them during regex transformations
        code_block_pattern = re.compile(r"(```[^\n]*\n.*?```)", re.DOTALL | re.MULTILINE)
        segments = code_block_pattern.split(text)

        cleaned_segments: list[str] = []
        for index, segment in enumerate(segments):
            # Odd segments represent fenced code blocks: preserve 100% verbatim
            if index % 2 == 1:
                cleaned_segments.append(segment)
                continue

            # Outside code blocks, apply aggressive boilerplate cleaning
            cleaned = segment
            cleaned = PAGE_NUMBER_PATTERN.sub("", cleaned)
            cleaned = CONFIDENTIAL_BANNER_PATTERN.sub("", cleaned)
            cleaned = EXTRA_BANNER_PATTERN.sub("", cleaned)
            cleaned = HTML_COMMENT_PATTERN.sub("", cleaned)
            cleaned = MULTIPLE_NEWLINES_PATTERN.sub("\n\n", cleaned)
            cleaned_segments.append(cleaned)

        return "".join(cleaned_segments).strip()

    def _partition_large_code_block(self, code_block: str, max_tokens: int) -> list[str]:
        """
        Partition a code block exceeding max_tokens into sub-chunks.

        Guarantees each sub-chunk retains valid matching opening and closing code fences.

        Args:
            code_block: Fenced code block string.
            max_tokens: Maximum token budget per code chunk.

        Returns:
            List of valid fenced code sub-chunks.
        """
        lines = code_block.splitlines()
        if not lines:
            return [code_block]

        opening_fence = lines[0] if lines[0].startswith("```") else "```"
        closing_fence = lines[-1] if lines[-1].startswith("```") else "```"

        body_lines = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        if not body_lines:
            return [code_block]

        fence_cost = count_tokens(opening_fence) + count_tokens(closing_fence) + 2
        effective_budget = max(50, max_tokens - fence_cost)

        sub_chunks: list[str] = []
        current_lines: list[str] = []
        current_tokens = 0

        for line in body_lines:
            line_cost = count_tokens(line) + 1
            if current_lines and (current_tokens + line_cost > effective_budget):
                sub_chunks.append(
                    f"{opening_fence}\n" + "\n".join(current_lines) + f"\n{closing_fence}"
                )
                current_lines = [line]
                current_tokens = line_cost
            else:
                current_lines.append(line)
                current_tokens += line_cost

        if current_lines:
            sub_chunks.append(
                f"{opening_fence}\n" + "\n".join(current_lines) + f"\n{closing_fence}"
            )

        return sub_chunks

    def _extract_atomic_units(self, text_segment: str) -> list[str]:
        """
        Deconstruct text segment into atomic semantic units (headings, tables, sentences).

        Preserves markdown tables, hex dumps, and headings as indivisible monolithic blocks.
        Deconstructs standard prose paragraphs into sentence-level atomic units to guarantee
        that chunk overlap accumulation can consistently satisfy the mandated 10-15% budget.

        Args:
            text_segment: Segment of markdown text without fenced code blocks.

        Returns:
            List of atomic string units.
        """
        paragraphs = text_segment.split("\n\n")
        units: list[str] = []

        for para in paragraphs:
            para_clean = para.strip()
            if not para_clean:
                continue

            lines = para_clean.split("\n")
            is_table = len(lines) >= 2 and all(
                line.strip().startswith("|") and line.strip().endswith("|")
                for line in lines
                if line.strip()
            )
            is_hex_dump = all(
                HEX_DUMP_ROW_PATTERN.match(line.strip()) for line in lines if line.strip()
            )

            # Tables and hex dumps are kept intact as single atomic blocks
            if is_table or is_hex_dump:
                units.append(para_clean)
                continue

            # Heading blocks and code blocks are kept intact as single atomic blocks
            if para_clean.startswith(("#", "```")):
                units.append(para_clean)
                continue

            # Standard prose paragraph: deconstruct into sentence-level atomic units
            sentences = [
                s.strip() for s in SENTENCE_SPLIT_PATTERN.split(para_clean) if s.strip()
            ]
            if not sentences:
                sentences = [para_clean]

            for sentence in sentences:
                sentence_tokens = count_tokens(sentence)
                if sentence_tokens <= self.default_chunk_size:
                    units.append(sentence)
                else:
                    # Fallback for single sentences exceeding chunk budget: split on words
                    words = sentence.split(" ")
                    step = max(1, int(self.default_chunk_size * 0.85))
                    for i in range(0, len(words), step):
                        chunk_words = words[i : i + self.default_chunk_size]
                        if chunk_words:
                            units.append(" ".join(chunk_words))

        return units

    def chunk_markdown(
        self,
        text: str,
        default_chunk_size: int | None = None,
        code_chunk_size: int | None = None,
        overlap: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextNode]:
        """
        Chunk markdown document into structurally-aware TextNodes.

        Enforces:
        - Default chunk size of 1024 tokens with 10-15% overlap.
        - Code blocks adaptively chunked to 256-512 tokens with intact code fences.
        - Preserves sentence boundaries and heading context.

        Args:
            text: Clean markdown document text.
            default_chunk_size: Optional override for default chunk size.
            code_chunk_size: Optional override for code block size.
            overlap: Optional override for chunk overlap.
            metadata: Optional base metadata dictionary to assign to nodes.

        Returns:
            List of generated TextNodes.

        Raises:
            ValueError: If overlap is greater than or equal to chunk size.
        """
        eff_chunk_size = default_chunk_size or self.default_chunk_size
        eff_code_chunk_size = code_chunk_size or self.code_chunk_size
        eff_overlap = overlap if overlap is not None else self.overlap

        if eff_overlap >= eff_chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        if eff_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")

        if not text or not text.strip():
            return []

        # 1. Balance unclosed code fences to prevent runaway state
        balanced_text = balance_code_fences(text)

        # 2. Break extremely long unbroken strings without whitespace
        safe_text = break_unbroken_strings(balanced_text)

        # 3. Partition into code block segments and text segments
        segments = CODE_BLOCK_PATTERN.split(safe_text)

        chunk_texts: list[str] = []

        for segment in segments:
            seg_clean = segment.strip()
            if not seg_clean:
                continue

            # Check if segment is a fenced code block
            if seg_clean.startswith("```"):
                code_tokens = count_tokens(seg_clean)
                if code_tokens <= eff_code_chunk_size:
                    chunk_texts.append(seg_clean)
                else:
                    sub_chunks = self._partition_large_code_block(
                        seg_clean, max_tokens=eff_code_chunk_size
                    )
                    chunk_texts.extend(sub_chunks)
            else:
                # Text segment: deconstruct into atomic units
                units = self._extract_atomic_units(seg_clean)
                if not units:
                    continue

                min_overlap = int(eff_chunk_size * DEFAULT_OVERLAP_RATIO_MIN)
                max_overlap = int(eff_chunk_size * DEFAULT_OVERLAP_RATIO_MAX)
                target_overlap = max(min_overlap, min(eff_overlap, max_overlap))

                current_units: list[str] = []
                current_tokens = 0

                for unit in units:
                    unit_tokens = count_tokens(unit)

                    if current_tokens + unit_tokens <= eff_chunk_size:
                        current_units.append(unit)
                        current_tokens += unit_tokens
                    else:
                        if current_units:
                            chunk_texts.append("\n\n".join(current_units).strip())

                        # Calculate overlap units for the next chunk targeting min_overlap to max_overlap window
                        overlap_units: list[str] = []
                        overlap_tokens = 0
                        if eff_overlap > 0:
                            for prev_unit in reversed(current_units):
                                cand_units = [prev_unit, *overlap_units]
                                cand_tokens = count_tokens("\n\n".join(cand_units))
                                if cand_tokens <= max_overlap:
                                    overlap_units.insert(0, prev_unit)
                                    overlap_tokens = cand_tokens
                                    if (
                                        overlap_tokens >= target_overlap
                                        and (overlap_tokens / eff_chunk_size)
                                        >= DEFAULT_OVERLAP_RATIO_MIN
                                    ):
                                        break
                                else:
                                    break

                            if (
                                eff_overlap > 0
                                and (overlap_tokens / eff_chunk_size)
                                < DEFAULT_OVERLAP_RATIO_MIN
                            ):
                                overlap_units = []

                        current_units = list(overlap_units)
                        current_units.append(unit)
                        current_tokens = count_tokens("\n\n".join(current_units))

                if current_units:
                    chunk_texts.append("\n\n".join(current_units).strip())

        # Construct TextNodes and assign metadata
        nodes: list[TextNode] = []
        for chunk_text in chunk_texts:
            clean_chunk = chunk_text.strip()
            if not clean_chunk:
                continue
            node = TextNode(text=clean_chunk)
            if metadata:
                node.metadata = dict(metadata)
                if all(k in node.metadata for k in STRICT_METADATA_KEYS):
                    sanitize_node_metadata(node)
            nodes.append(node)

        return nodes

    def ingest_file(self, file_path: Path | str) -> list[TextNode]:
        """
        Ingest a single document file into sanitized TextNodes.

        Extracts metadata, performs conditional standard or OCR extraction,
        removes boilerplate, chunks content adaptively, and sanitizes node metadata.

        Args:
            file_path: Path to the document.

        Returns:
            List of sanitized TextNodes with exact 5-field metadata.
        """
        path = Path(file_path)
        strict_meta = extract_document_metadata(path)
        meta_dict = strict_meta.to_dict()

        raw_text = self.extract_text(path)
        if not raw_text.strip():
            return []

        cleaned_text = self.clean_boilerplate(raw_text)
        if not cleaned_text.strip():
            return []

        nodes = self.chunk_markdown(
            text=cleaned_text,
            default_chunk_size=self.default_chunk_size,
            code_chunk_size=self.code_chunk_size,
            overlap=self.overlap,
            metadata=meta_dict,
        )

        for node in nodes:
            node.metadata = dict(meta_dict)
            sanitize_node_metadata(node)

        return nodes

    def ingest_directory(
        self,
        dir_path: Path | str,
        recursive: bool = True,
        supported_extensions: tuple[str, ...] = (
            ".pdf",
            ".txt",
            ".md",
            ".markdown",
            ".png",
            ".jpg",
            ".jpeg",
        ),
    ) -> list[TextNode]:
        """
        Ingest all supported document files in a directory.

        Detects and skips exact duplicate files based on SHA-256 hash.

        Args:
            dir_path: Target directory path.
            recursive: Whether to traverse subdirectories.
            supported_extensions: Tuple of allowed file extensions.

        Returns:
            Aggregated list of sanitized TextNodes.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If dir_path is not a directory.
        """
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Target path is not a directory: {path}")

        pattern = "**/*" if recursive else "*"
        all_nodes: list[TextNode] = []
        indexed_hashes: set[str] = set()

        for candidate in sorted(path.glob(pattern)):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in supported_extensions:
                continue

            try:
                file_hash = compute_file_hash(candidate)
                if file_hash in indexed_hashes:
                    logger.info(
                        f"Skipping duplicate file in directory: {candidate.name} (hash: {file_hash})"
                    )
                    continue
                indexed_hashes.add(file_hash)

                nodes = self.ingest_file(candidate)
                all_nodes.extend(nodes)
            except (DocumentParsingError, CorruptedDocumentError) as parse_err:
                logger.warning(f"Skipping unparseable document {candidate.name}: {parse_err}")
            except (RuntimeError, ValueError, OSError, TypeError) as exc:
                logger.error(f"Unexpected error ingesting {candidate.name}: {exc}")

        return all_nodes


# Global default pipeline instance for module-level functions
_default_pipeline: DoclingPipeline | None = None


def get_default_pipeline() -> DoclingPipeline:
    """
    Retrieve or initialize the default global DoclingPipeline instance.

    Returns:
        Singleton DoclingPipeline instance.
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = DoclingPipeline()
    return _default_pipeline


def ingest_file(file_path: Path | str) -> list[TextNode]:
    """
    Ingest a single document file using the default pipeline.

    Args:
        file_path: Path to the target document.

    Returns:
        List of sanitized TextNodes with exact 5-field metadata.
    """
    return get_default_pipeline().ingest_file(file_path)


def ingest_directory(
    dir_path: Path | str,
    recursive: bool = True,
) -> list[TextNode]:
    """
    Ingest all documents in a directory using the default pipeline.

    Args:
        dir_path: Target directory path.
        recursive: Whether to scan subdirectories.

    Returns:
        Aggregated list of sanitized TextNodes.
    """
    return get_default_pipeline().ingest_directory(dir_path, recursive=recursive)


def chunk_markdown(
    text: str,
    default_chunk_size: int = DEFAULT_CHUNK_SIZE,
    code_chunk_size: int = DEFAULT_CODE_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextNode]:
    """
    Chunk markdown text into TextNodes using structural and adaptive rules.

    Args:
        text: Input markdown text.
        default_chunk_size: Chunk size in tokens for general text.
        code_chunk_size: Chunk size in tokens for code blocks.
        overlap: Overlap token budget between chunks.

    Returns:
        List of generated TextNodes.
    """
    pipeline = DoclingPipeline(
        default_chunk_size=default_chunk_size,
        code_chunk_size=code_chunk_size,
        overlap=overlap,
    )
    return pipeline.chunk_markdown(text)
