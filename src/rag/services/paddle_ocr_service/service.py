"""
PaddleOCR Service implementation.

Provides OCR functionality using PaddleOCR to extract text and code from images.
"""

import logging
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from pathlib import Path
import numpy as np
from PIL import Image
import io
import base64

if TYPE_CHECKING:
    # Type hints only - actual import happens at runtime
    from paddleocr import PaddleOCR  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class PaddleOCRService:
    """Service for extracting text from images using PaddleOCR."""
    
    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        use_angle_cls: bool = True,
        show_log: bool = False,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        cls_model_dir: Optional[str] = None,
        use_space_char: bool = True,
        enable_mkldnn: bool = False,
        cpu_threads: int = 10,
        min_confidence: float = 0.5
    ) -> None:
        """
        Initialize PaddleOCR service.
        
        Args:
            lang: OCR language (en, ch, fr, german, korean, japan)
            use_gpu: Use GPU acceleration
            use_angle_cls: Enable text angle classification
            show_log: Show PaddleOCR logs
            det_model_dir: Path to custom detection model
            rec_model_dir: Path to custom recognition model
            cls_model_dir: Path to custom classification model
            use_space_char: Recognize space characters
            enable_mkldnn: Enable MKLDNN acceleration
            cpu_threads: Number of CPU threads
            min_confidence: Minimum confidence threshold
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.show_log = show_log
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.cls_model_dir = cls_model_dir
        self.use_space_char = use_space_char
        self.enable_mkldnn = enable_mkldnn
        self.cpu_threads = cpu_threads
        self.min_confidence = min_confidence
        
        self.ocr: Any = None  # Will be initialized in _initialize_ocr()
        self._initialize_ocr()
        # After _initialize_ocr(), self.ocr is guaranteed to be a PaddleOCR instance
        # (or __init__ will raise an exception)
    
    def _initialize_ocr(self) -> None:
        """Initialize PaddleOCR engine."""
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-not-found]
            
            # Build OCR configuration
            ocr_config = {
                'lang': self.lang,
                'use_gpu': self.use_gpu,
                'use_angle_cls': self.use_angle_cls,
                'show_log': self.show_log,
                'use_space_char': self.use_space_char,
                'enable_mkldnn': self.enable_mkldnn,
                'cpu_threads': self.cpu_threads,
            }
            
            # Add custom model paths if provided
            if self.det_model_dir:
                ocr_config['det_model_dir'] = self.det_model_dir
            if self.rec_model_dir:
                ocr_config['rec_model_dir'] = self.rec_model_dir
            if self.cls_model_dir:
                ocr_config['cls_model_dir'] = self.cls_model_dir
            
            self.ocr = PaddleOCR(**ocr_config)
            logger.info(f"PaddleOCR initialized with language: {self.lang}, GPU: {self.use_gpu}")
            
        except ImportError:
            logger.error("PaddleOCR not installed. Please install with: pip install paddleocr paddlepaddle")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    def extract_text_from_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
        return_confidence: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Extract text from a single image.
        
        Args:
            image: Image input (file path, PIL Image, numpy array, or bytes)
            return_confidence: Return confidence scores along with text
            
        Returns:
            Extracted text string or dict with text and confidence scores
        """
        # Convert image to appropriate format
        img_array = self._prepare_image(image)
        
        try:
            # Perform OCR (self.ocr is guaranteed to be initialized in __init__)
            result = self.ocr.ocr(img_array, cls=self.use_angle_cls)
            
            if not result or not result[0]:
                logger.warning("No text detected in image")
                return "" if not return_confidence else {"text": "", "confidence": 0.0}
            
            # Extract text and confidence scores
            extracted_lines = []
            total_confidence = 0.0
            valid_detections = 0
            
            for line in result[0]:
                if len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # Filter by minimum confidence
                    if confidence >= self.min_confidence:
                        extracted_lines.append(text)
                        total_confidence += confidence
                        valid_detections += 1
            
            # Combine extracted text
            full_text = "\n".join(extracted_lines)
            avg_confidence = total_confidence / valid_detections if valid_detections > 0 else 0.0
            
            logger.info(f"Extracted {len(extracted_lines)} text lines with avg confidence: {avg_confidence:.2f}")
            
            if return_confidence:
                return {
                    "text": full_text,
                    "confidence": avg_confidence,
                    "lines_count": len(extracted_lines)
                }
            
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from image: {e}")
            return "" if not return_confidence else {"text": "", "confidence": 0.0}
    
    def extract_text_from_images(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray, bytes]],
        return_confidence: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Extract text from multiple images.
        
        Args:
            images: List of image inputs
            return_confidence: Return confidence scores along with text
            
        Returns:
            List of extracted text strings or dicts with text and confidence
        """
        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            result = self.extract_text_from_image(image, return_confidence=return_confidence)
            results.append(result)
        
        return results
    
    def _prepare_image(self, image: Union[str, Path, Image.Image, np.ndarray, bytes]) -> np.ndarray:
        """
        Prepare image for OCR processing.
        
        Args:
            image: Image input in various formats
            
        Returns:
            Image as numpy array
        """
        # If already numpy array, return as is
        if isinstance(image, np.ndarray):
            return image
        
        # If PIL Image, convert to numpy
        if isinstance(image, Image.Image):
            return np.array(image)
        
        # If bytes (e.g., from base64), decode to PIL then numpy
        if isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image))
            return np.array(pil_image)
        
        # If string or Path, load image file
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            pil_image = Image.open(image_path)
            return np.array(pil_image)
        
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    def extract_text_from_base64(self, base64_string: str, return_confidence: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Extract text from base64-encoded image.
        
        Args:
            base64_string: Base64-encoded image string
            return_confidence: Return confidence scores along with text
            
        Returns:
            Extracted text string or dict with text and confidence
        """
        try:
            # Remove data URI prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',', 1)[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)
            
            return self.extract_text_from_image(image_bytes, return_confidence=return_confidence)
            
        except Exception as e:
            logger.error(f"Failed to extract text from base64 image: {e}")
            return "" if not return_confidence else {"text": "", "confidence": 0.0}
    
    def is_code_image(self, extracted_text: str) -> bool:
        """
        Heuristic to determine if extracted text is likely code.
        
        Args:
            extracted_text: Text extracted from image
            
        Returns:
            True if text appears to be code
        """
        # Check for common code indicators
        code_indicators = [
            '{', '}', '(', ')', '[', ']',  # Brackets
            'def ', 'class ', 'function ', 'var ', 'const ', 'let ',  # Keywords
            '=>', '->', '==', '!=', '<=', '>=',  # Operators
            'import ', 'from ', 'include ', '#include',  # Imports
            'public ', 'private ', 'protected ',  # Access modifiers
            '//', '/*', '*/', '#',  # Comments
        ]
        
        # Count code indicators
        indicator_count = sum(1 for indicator in code_indicators if indicator in extracted_text)
        
        # If more than 3 indicators, likely code
        return indicator_count >= 3
    
    def format_extracted_text(
        self,
        text: str,
        format_type: str = "markdown",
        is_code: bool = False
    ) -> str:
        """
        Format extracted text for insertion into markdown.
        
        Args:
            text: Extracted text
            format_type: Output format (markdown, html, plain)
            is_code: Whether text is code
            
        Returns:
            Formatted text string
        """
        if not text.strip():
            return ""
        
        if format_type == "markdown":
            if is_code:
                # Wrap in code block
                return f"\n```\n{text}\n```\n"
            else:
                # Regular text, add newlines
                return f"\n{text}\n"
        
        elif format_type == "html":
            if is_code:
                return f"\n<pre><code>{text}</code></pre>\n"
            else:
                return f"\n<p>{text}</p>\n"
        
        else:  # plain
            return f"\n{text}\n"
