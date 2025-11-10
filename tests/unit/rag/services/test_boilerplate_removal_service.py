"""
Unit tests for boilerplate removal service.

Tests aggressive boilerplate removal while preserving technical content.
"""

from src.rag.services.boilerplate_removal_service import (
    BoilerplateRemovalService,
    remove_boilerplate,
    remove_html_comments,
    remove_blog_metadata,
    remove_footer_timestamps,
)


class TestBoilerplateRemovalService:
    """Test suite for BoilerplateRemovalService class."""
    
    def test_remove_html_comments(self) -> None:
        """Test removal of HTML comments."""
        text = """
        <!-- image -->
        This is actual content.
        <!-- another comment -->
        More content here.
        """
        
        cleaned = remove_html_comments(text)
        
        assert "<!-- image -->" not in cleaned
        assert "<!-- another comment -->" not in cleaned
        assert "This is actual content." in cleaned
        assert "More content here." in cleaned
    
    def test_remove_blog_metadata(self) -> None:
        """Test removal of blog-specific metadata."""
        text = """
        devblogs.microsoft.com /oldnewthing/20030721-00
        Raymond Chen Follow
        This is the actual article content.
        Follow
        """
        
        cleaned = remove_blog_metadata(text)
        
        assert "devblogs.microsoft.com" not in cleaned
        assert "Raymond Chen Follow" not in cleaned
        assert "This is the actual article content." in cleaned
    
    def test_remove_footer_timestamps(self) -> None:
        """Test removal of date footers."""
        text = """This is article content.
July 21, 2003"""
        
        cleaned = remove_footer_timestamps(text)
        
        assert "July 21, 2003" not in cleaned
        assert "This is article content." in cleaned
    
    def test_preserve_code_blocks(self) -> None:
        """Test that code blocks are preserved during cleaning."""
        text = """
        <!-- image -->
        Here is some code:
        ```python
        def example():
            # This comment should stay
            return True
        ```
        Raymond Chen Follow
        """
        
        service = BoilerplateRemovalService(aggressive_mode=True)
        cleaned = service.remove_boilerplate(text)
        
        # Boilerplate removed
        assert "<!-- image -->" not in cleaned
        assert "Raymond Chen Follow" not in cleaned
        
        # Code block preserved
        assert "```python" in cleaned
        assert "def example():" in cleaned
        assert "# This comment should stay" in cleaned
        assert "return True" in cleaned
    
    def test_preserve_technical_content(self) -> None:
        """Test preservation of technical terms and function names."""
        text = """
        <!-- image -->
        The WM_NCCALCSIZE message is important.
        Use SubclassWindow for this.
        Raymond Chen Follow
        """
        
        cleaned = remove_boilerplate(text)
        
        # Boilerplate removed
        assert "<!-- image -->" not in cleaned
        assert "Raymond Chen Follow" not in cleaned
        
        # Technical content preserved
        assert "WM_NCCALCSIZE" in cleaned
        assert "SubclassWindow" in cleaned
    
    def test_preserve_markdown_structure(self) -> None:
        """Test preservation of markdown headers and structure."""
        text = """
        <!-- image -->
        ## Main Heading
        
        Some content here.
        
        ### Subheading
        
        More content.
        Raymond Chen Follow
        """
        
        cleaned = remove_boilerplate(text)
        
        # Markdown preserved
        assert "## Main Heading" in cleaned
        assert "### Subheading" in cleaned
        assert "Some content here." in cleaned
        
        # Boilerplate removed
        assert "<!-- image -->" not in cleaned
        assert "Raymond Chen Follow" not in cleaned
    
    def test_real_world_example(self) -> None:
        """Test with real-world example from Vx Underground collection."""
        text = """<!-- image --> 
devblogs.microsoft.com /oldnewthing/20030721-00 
Raymond Chen <!-- image --> 
## What are you talking about? 
Tweak UI is part of the Windows XP PowerToys. It was recently updated to version 2.10. 

## What OS is required? 
Windows XP Service Pack 1 (or higher) or Windows Server 2003 (all versions). Not supported: Windows XP RTM, Windows 2000, NT 4, Windows 95, 98, or Me. 

## Why didn't you make a big announcement? 
I didn't find out about it until somebody told me. I don't control the announcements page. I figured word would get out - and it did. 

## Why is the new Tweak UI so much smaller than the old one? 
Leaner installer program. 

## Are you always this terse? 
I'm not sure yet; I'm new at this. 
Raymond Chen Follow
July 21, 2003 
<!-- image -->"""
        
        cleaned = remove_boilerplate(text, aggressive_mode=True)
        
        # Verify boilerplate removed
        assert "<!-- image -->" not in cleaned
        assert "devblogs.microsoft.com" not in cleaned
        assert "Raymond Chen Follow" not in cleaned
        assert "July 21, 2003" not in cleaned
        
        # Verify content preserved
        assert "## What are you talking about?" in cleaned
        assert "Tweak UI" in cleaned
        assert "Windows XP PowerToys" in cleaned
        assert "## What OS is required?" in cleaned
        assert "Windows XP Service Pack 1" in cleaned
    
    def test_aggressive_mode_vs_normal(self) -> None:
        """Test difference between aggressive and normal mode."""
        text = """
        <!-- image -->
        Home > Blog > Articles
        Main content here.
        Raymond Chen
        """
        
        normal = BoilerplateRemovalService(aggressive_mode=False)
        aggressive = BoilerplateRemovalService(aggressive_mode=True)
        
        normal_cleaned = normal.remove_boilerplate(text)
        aggressive_cleaned = aggressive.remove_boilerplate(text)
        
        # Both should remove HTML comments
        assert "<!-- image -->" not in normal_cleaned
        assert "<!-- image -->" not in aggressive_cleaned
        
        # Aggressive should remove more
        assert len(aggressive_cleaned) <= len(normal_cleaned)
    
    def test_empty_text(self) -> None:
        """Test handling of empty text."""
        assert remove_boilerplate("") == ""
        assert remove_boilerplate("   ") == ""
    
    def test_statistics_tracking(self) -> None:
        """Test that statistics are tracked correctly."""
        text = """
        <!-- image -->
        Content here.
        Raymond Chen Follow
        """
        
        service = BoilerplateRemovalService()
        service.remove_boilerplate(text)
        stats = service.get_statistics()
        
        assert isinstance(stats, dict)
        assert len(stats) > 0  # Should have removed something
    
    def test_whitespace_cleanup(self) -> None:
        """Test that excessive whitespace is cleaned up."""
        text = """


        Content here.
        
        
        
        More content.
        
        
        """
        
        cleaned = remove_boilerplate(text)
        
        # Should have at most double newlines (paragraph breaks)
        assert "\n\n\n" not in cleaned
        # Should be trimmed
        assert not cleaned.startswith("\n")
        assert not cleaned.endswith("\n\n\n")
    
    def test_preserve_lists(self) -> None:
        """Test preservation of numbered and bulleted lists."""
        text = """
        <!-- image -->
        Here are the options:
        1. First option
        2. Second option
        * Bullet point
        - Another bullet
        Raymond Chen Follow
        """
        
        cleaned = remove_boilerplate(text)
        
        # Lists preserved
        assert "1. First option" in cleaned
        assert "2. Second option" in cleaned
        assert "* Bullet point" in cleaned
        assert "- Another bullet" in cleaned
        
        # Boilerplate removed
        assert "<!-- image -->" not in cleaned
        assert "Raymond Chen Follow" not in cleaned


class TestPatternSpecificFunctions:
    """Test individual pattern removal functions."""
    
    def test_remove_html_comments_only(self) -> None:
        """Test HTML comment removal doesn't affect other content."""
        text = "<!-- image --> Content Raymond Chen Follow"
        cleaned = remove_html_comments(text)
        
        assert "<!-- image -->" not in cleaned
        assert "Raymond Chen Follow" in cleaned  # Should still be there
    
    def test_remove_blog_metadata_only(self) -> None:
        """Test blog metadata removal doesn't affect other content."""
        text = "<!-- image --> Content Raymond Chen Follow"
        cleaned = remove_blog_metadata(text)
        
        assert "Raymond Chen Follow" not in cleaned
        assert "<!-- image -->" in cleaned  # Should still be there
    
    def test_multiple_html_comments(self) -> None:
        """Test removal of multiple HTML comments."""
        text = """
        <!-- image -->
        Content
        <!-- another -->
        More
        <!-- final -->
        """
        
        cleaned = remove_html_comments(text)
        
        assert "<!--" not in cleaned
        assert "-->" not in cleaned
        assert "Content" in cleaned
        assert "More" in cleaned
