"""
Boilerplate removal service for aggressive cleaning of web artifacts and document noise.

This service removes boilerplate content from parsed documents AFTER OCR/parsing stage
but BEFORE normalization. It handles web artifacts, navigation elements, footers,
and other non-content patterns specific to the Vx Underground collection.
"""

from .service import (
    BoilerplateRemovalService,
    remove_boilerplate,
    remove_html_comments,
    remove_blog_metadata,
    remove_footer_timestamps,
)

__all__ = [
    'BoilerplateRemovalService',
    'remove_boilerplate',
    'remove_html_comments',
    'remove_blog_metadata',
    'remove_footer_timestamps',
]
