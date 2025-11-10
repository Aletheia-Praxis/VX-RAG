"""
Boilerplate removal service for aggressive cleaning of web artifacts and document noise.

This service removes boilerplate content from parsed documents AFTER OCR/parsing stage
but BEFORE normalization. It handles web artifacts, navigation elements, footers,
and other non-content patterns specific to the Vx Underground collection.

Pipeline position:
    Docling/OCR → PaddleOCR (needs <!-- image -->) → BoilerplateRemovalService → normalize_text()
"""

import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BoilerplatePattern:
    """Pattern configuration for boilerplate removal."""
    
    name: str
    pattern: str
    flags: int = re.IGNORECASE | re.MULTILINE
    description: str = ""
    enabled: bool = True


class BoilerplateRemovalService:
    """
    Aggressive boilerplate removal service for technical blog posts and documents.
    
    Designed specifically for Vx Underground collection (Raymond Chen's "The Old New Thing" blog).
    Removes web artifacts, HTML comments, navigation elements, and footers while preserving
    technical content, code blocks, and document structure.
    """
    
    def __init__(self, aggressive_mode: bool = True) -> None:
        """
        Initialize boilerplate removal service.
        
        Args:
            aggressive_mode: If True, uses aggressive patterns. If False, only removes obvious boilerplate.
        """
        self.aggressive_mode = aggressive_mode
        self.patterns = self._build_patterns()
        self.stats: Dict[str, int] = {}
        
    def _build_patterns(self) -> List[BoilerplatePattern]:
        """
        Build list of boilerplate patterns to remove.
        
        Returns:
            List of BoilerplatePattern objects
        """
        patterns = [
            # HTML artifacts (always remove)
            BoilerplatePattern(
                name="html_comments",
                pattern=r'<!--\s*[^>]*\s*-->',
                description="HTML comments like <!-- image -->"
            ),
            
            # Web navigation and metadata
            BoilerplatePattern(
                name="blog_urls",
                pattern=r'devblogs\.microsoft\.com\s*/oldnewthing/\d{8}-\d+',
                description="Blog URL patterns"
            ),
            
            BoilerplatePattern(
                name="author_follow",
                pattern=r'Raymond\s+Chen\s+Follow\s*',
                description="Author name with Follow button"
            ),
            
            BoilerplatePattern(
                name="standalone_follow",
                pattern=r'^\s*Follow\s*$',
                description="Standalone Follow button"
            ),
            
            # Footer timestamps (isolated dates at end or between sections)
            BoilerplatePattern(
                name="footer_date",
                pattern=r'(?:^|\n)\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\s*(?:\n|$)',
                description="Date timestamps as footers"
            ),
            
            # Standalone author name (only if in last few lines and isolated)
            BoilerplatePattern(
                name="standalone_author",
                pattern=r'\n\s*Raymond\s+Chen\s*\n',
                description="Standalone author name",
                enabled=self.aggressive_mode
            ),
        ]
        
        # Additional aggressive patterns
        if self.aggressive_mode:
            patterns.extend([
                # Empty lines with only HTML artifacts
                BoilerplatePattern(
                    name="empty_artifact_lines",
                    pattern=r'^\s*(?:<!--[^>]*-->|\s)+$',
                    flags=re.MULTILINE,
                    description="Lines with only HTML comments/whitespace"
                ),
                
                # Navigation breadcrumbs
                BoilerplatePattern(
                    name="breadcrumbs",
                    pattern=r'^\s*(?:Home|Blog|Archives?|Categories?)\s*[>›/]\s*.*$',
                    flags=re.MULTILINE | re.IGNORECASE,
                    description="Navigation breadcrumbs"
                ),
                
                # Social sharing elements
                BoilerplatePattern(
                    name="social_sharing",
                    pattern=r'(?:Share|Tweet|Like|Follow|Subscribe)\s+(?:on|via)\s+(?:Twitter|Facebook|LinkedIn|Reddit)',
                    description="Social media sharing elements"
                ),
            ])
        
        return patterns
    
    def remove_boilerplate(self, text: str) -> str:
        """
        Remove boilerplate from text while preserving technical content.
        
        Args:
            text: Input text with potential boilerplate
            
        Returns:
            Cleaned text with boilerplate removed
        """
        if not text:
            return text
            
        original_length = len(text)
        cleaned_text = text
        self.stats = {}
        
        # Protect code blocks from modification
        code_blocks, text_with_placeholders = self._protect_code_blocks(cleaned_text)
        
        # Apply each pattern
        for pattern_config in self.patterns:
            if not pattern_config.enabled:
                continue
                
            before_length = len(text_with_placeholders)
            text_with_placeholders = re.sub(
                pattern_config.pattern,
                '',
                text_with_placeholders,
                flags=pattern_config.flags
            )
            after_length = len(text_with_placeholders)
            
            removed = before_length - after_length
            if removed > 0:
                self.stats[pattern_config.name] = removed
                logger.debug(
                    f"Pattern '{pattern_config.name}' removed {removed} characters: "
                    f"{pattern_config.description}"
                )
        
        # Restore code blocks
        cleaned_text = self._restore_code_blocks(text_with_placeholders, code_blocks)
        
        # Clean up excessive whitespace (but preserve paragraph breaks)
        cleaned_text = self._cleanup_whitespace(cleaned_text)
        
        final_length = len(cleaned_text)
        total_removed = original_length - final_length
        
        if total_removed > 0:
            removal_percentage = (total_removed / original_length) * 100
            logger.info(
                f"Removed {total_removed} characters ({removal_percentage:.1f}%) of boilerplate"
            )
        
        return cleaned_text
    
    def _protect_code_blocks(self, text: str) -> tuple[List[str], str]:
        """
        Extract and protect code blocks from modification.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (list of code blocks, text with placeholders)
        """
        code_blocks: List[str] = []
        placeholder_template = "<<<CODE_BLOCK_{}>>>"
        
        # Pattern for markdown code blocks (```...```)
        code_pattern = r'```[\s\S]*?```'
        
        def replace_with_placeholder(match: re.Match[str]) -> str:
            index = len(code_blocks)
            code_blocks.append(match.group(0))
            return placeholder_template.format(index)
        
        text_with_placeholders = re.sub(code_pattern, replace_with_placeholder, text)
        
        return code_blocks, text_with_placeholders
    
    def _restore_code_blocks(self, text: str, code_blocks: List[str]) -> str:
        """
        Restore protected code blocks.
        
        Args:
            text: Text with placeholders
            code_blocks: List of original code blocks
            
        Returns:
            Text with restored code blocks
        """
        for i, code_block in enumerate(code_blocks):
            placeholder = f"<<<CODE_BLOCK_{i}>>>"
            text = text.replace(placeholder, code_block)
        
        return text
    
    def _cleanup_whitespace(self, text: str) -> str:
        """
        Clean up excessive whitespace while preserving document structure.
        
        Args:
            text: Input text
            
        Returns:
            Text with cleaned whitespace
        """
        # Remove trailing whitespace from lines
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        
        # Replace multiple blank lines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from entire document
        text = text.strip()
        
        return text
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about removed boilerplate.
        
        Returns:
            Dictionary with pattern names and removed character counts
        """
        return self.stats.copy()


def remove_boilerplate(
    text: str,
    aggressive_mode: bool = True
) -> str:
    """
    Convenience function to remove boilerplate from text.
    
    Args:
        text: Input text with potential boilerplate
        aggressive_mode: If True, uses aggressive removal patterns
        
    Returns:
        Cleaned text with boilerplate removed
    """
    service = BoilerplateRemovalService(aggressive_mode=aggressive_mode)
    return service.remove_boilerplate(text)


# Pattern-specific removal functions for fine-grained control

def remove_html_comments(text: str) -> str:
    """Remove HTML comments from text."""
    return re.sub(r'<!--\s*[^>]*\s*-->', '', text)


def remove_blog_metadata(text: str) -> str:
    """Remove blog-specific metadata (URLs, author, dates)."""
    patterns = [
        r'devblogs\.microsoft\.com\s*/oldnewthing/\d{8}-\d+',
        r'Raymond\s+Chen\s+Follow',
        r'^\s*Follow\s*$',
    ]
    
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    return cleaned


def remove_footer_timestamps(text: str) -> str:
    """Remove timestamp footers at document end."""
    pattern = r'\n\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\s*$'
    return re.sub(pattern, '', text, flags=re.IGNORECASE)
