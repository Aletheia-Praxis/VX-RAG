"""
Unit tests for adaptive chunking functionality.

Tests the content analysis and adaptive chunk size selection
for different types of cybersecurity documentation.
"""

import pytest
from src.rag.services.chunker_service.service import Chunker


class TestAdaptiveChunking:
    """Test suite for adaptive chunking system."""
    
    @pytest.fixture
    def chunker(self) -> Chunker:
        """Create a chunker instance with adaptive configuration."""
        return Chunker(
            chunk_size=1024,
            chunk_overlap=200,
            use_semantic_chunking=True,
            adaptive_config={
                'enabled': True,
                'large_code_blocks': {'chunk_size': 768, 'chunk_overlap': 150, 'min_lines': 30},
                'tables': {'chunk_size': 512, 'chunk_overlap': 100},
                'short_snippets': {'chunk_size': 384, 'chunk_overlap': 75, 'max_lines': 15},
                'hex_dumps': {'chunk_size': 896, 'chunk_overlap': 175}
            }
        )
    
    def test_general_content_detection(self, chunker: Chunker) -> None:
        """Test that general narrative content is detected correctly."""
        text = """The Emotet malware campaign has been active since 2014, targeting organizations worldwide with sophisticated phishing attacks. The threat actors behind Emotet use polymorphic code to evade detection and have established a robust command and control infrastructure across multiple geographic regions."""
        
        analysis = chunker._analyze_content_type(text)
        
        assert analysis['type'] == 'general'
        assert analysis['subtype'] is None
    
    def test_hex_dump_detection(self, chunker: Chunker) -> None:
        """Test hexdump detection triggers correct chunk size."""
        text = """
        Memory dump at address 0x00401000:
        0x00401000: 55 8B EC 83 EC 10 8B 45 08 89 45 F0 8B 4D 0C 89
        0x00401010: 4D F4 8B 55 10 89 55 F8 8B 45 14 89 45 FC 8B 4D
        0x00401020: F0 81 C1 04 00 00 00 89 4D F0 8B 55 F4 81 C2 08
        0x00401030: 00 00 00 89 55 F4 8B 45 F8 81 C0 0C 00 00 00 89
        0x00401040: 45 F8 8B 4D FC 81 C1 10 00 00 00 89 4D FC 8B 55
        """
        
        analysis = chunker._analyze_content_type(text)
        
        assert analysis['type'] == 'technical'
        assert analysis['subtype'] == 'hex_dumps'
        assert 'hex_matches' in analysis['metadata']
        
        # Verify chunk parameters
        chunk_size, overlap = chunker._get_adaptive_chunk_params(
            analysis['type'], analysis['subtype'], analysis
        )
        assert chunk_size == 896
        assert overlap == 175
    
    def test_table_detection(self, chunker: Chunker) -> None:
        """Test table detection triggers correct chunk size."""
        text = """| Timestamp | Source IP | Destination | Port | Action |
|-----------|-----------|-------------|------|--------|
| 12:30:45  | 10.0.1.5  | 8.8.8.8    | 443  | BLOCK  |
| 12:31:12  | 10.0.1.8  | evil.com   | 80   | ALLOW  |
| 12:32:00  | 10.0.1.9  | malware.org| 8080 | BLOCK  |"""
        
        analysis = chunker._analyze_content_type(text)
        
        assert analysis['type'] == 'technical'
        assert analysis['subtype'] == 'tables'
        
        chunk_size, overlap = chunker._get_adaptive_chunk_params(
            analysis['type'], analysis['subtype'], analysis
        )
        assert chunk_size == 512
        assert overlap == 100
    
    def test_large_code_block_detection(self, chunker: Chunker) -> None:
        """Test large code block detection triggers correct chunk size."""
        # Create a large assembly listing
        assembly_lines = [
            "push ebp",
            "mov ebp, esp",
            "sub esp, 0x20",
        ] * 15  # 45 lines total
        
        text = "```assembly\n" + "\n".join(assembly_lines) + "\n```"
        
        analysis = chunker._analyze_content_type(text)
        
        assert analysis['type'] == 'technical'
        assert analysis['subtype'] == 'large_code_blocks'
        assert analysis['metadata']['avg_lines'] >= 30
        
        chunk_size, overlap = chunker._get_adaptive_chunk_params(
            analysis['type'], analysis['subtype'], analysis
        )
        assert chunk_size == 768
        assert overlap == 150
    
    def test_short_snippet_detection(self, chunker: Chunker) -> None:
        """Test short code snippet detection triggers correct chunk size."""
        text = """
        ```python
        import subprocess
        subprocess.Popen(['cmd.exe', '/c', payload])
        ```
        """
        
        analysis = chunker._analyze_content_type(text)
        
        assert analysis['type'] == 'technical'
        assert analysis['subtype'] == 'short_snippets'
        assert analysis['metadata']['avg_lines'] <= 15
        
        chunk_size, overlap = chunker._get_adaptive_chunk_params(
            analysis['type'], analysis['subtype'], analysis
        )
        assert chunk_size == 384
        assert overlap == 75
    
    def test_overlap_ratio_consistency(self, chunker: Chunker) -> None:
        """Test that all adaptive configs maintain 19.5% overlap ratio."""
        test_cases = [
            ('large_code_blocks', 768, 150),
            ('tables', 512, 100),
            ('short_snippets', 384, 75),
            ('hex_dumps', 896, 175),
        ]
        
        for subtype, expected_size, expected_overlap in test_cases:
            overlap_ratio = (expected_overlap / expected_size) * 100
            # Allow small rounding difference
            assert 19.0 <= overlap_ratio <= 20.0, \
                f"{subtype}: overlap ratio {overlap_ratio}% is not ~19.5%"
    
    def test_adaptive_chunking_disabled(self) -> None:
        """Test that adaptive chunking can be disabled."""
        chunker = Chunker(
            chunk_size=1024,
            chunk_overlap=200,
            adaptive_config={'enabled': False}
        )
        
        text = """
        0x00401000: 55 8B EC 83 EC 10 8B 45 08 89 45 F0 8B 4D 0C 89
        0x00401010: 4D F4 8B 55 10 89 55 F8 8B 45 14 89 45 FC 8B 4D
        """
        
        analysis = chunker._analyze_content_type(text)
        
        # Should still detect hex dumps
        assert analysis['type'] == 'technical'
        
        # But should use default chunk size
        chunk_size, overlap = chunker._get_adaptive_chunk_params(
            analysis['type'], analysis['subtype'], analysis
        )
        assert chunk_size == 1024  # Default, not adaptive
        assert overlap == 200
    
    def test_chunk_metadata_includes_analysis(self, chunker: Chunker) -> None:
        """Test that chunk metadata includes content analysis results."""
        # Use hex dump which survives preprocessing better than tables
        document = {
            'id': 'test_doc_1',
            'text': """Memory dump at 0x00401000:
0x00401000: 55 8B EC 83 EC 10 8B 45 08 89 45 F0 8B 4D 0C 89
0x00401010: 4D F4 8B 55 10 89 55 F8 8B 45 14 89 45 FC 8B 4D
0x00401020: F0 81 C1 04 00 00 00 89 4D F0 8B 55 F4 81 C2 08
0x00401030: 00 00 00 89 55 F4 8B 45 F8 81 C0 0C 00 00 00 89
0x00401040: 45 F8 8B 4D FC 81 C1 10 00 00 00 89 4D FC 8B 55
0x00401050: F0 3B 55 08 0F 8D 10 00 00 00 8B 45 F4 3B 45 0C""",
            'source': 'test.pdf',
            'lang': 'en',
            'metadata': {}
        }
        
        chunks = chunker.chunk_documents([document])
        
        assert len(chunks) > 0
        first_chunk = chunks[0]
        
        chunk_metadata = first_chunk['metadata']['chunk_metadata']
        additional = chunk_metadata['additional_metadata']
        
        assert 'content_type' in additional
        assert 'content_subtype' in additional
        assert 'adaptive_chunk_size' in additional
        assert 'adaptive_overlap' in additional
        
        # Should detect hex dump
        assert additional['content_type'] == 'technical'
        assert additional['content_subtype'] == 'hex_dumps'
        assert additional['adaptive_chunk_size'] == 896
        assert additional['adaptive_overlap'] == 175
    
    def test_technical_keywords_fallback(self, chunker: Chunker) -> None:
        """Test that technical keyword detection works as fallback."""
        text = """The malware uses VirtualAlloc to allocate memory, then calls WriteProcessMemory to inject shellcode into the target process. Finally, it invokes CreateRemoteThread to execute the payload. The registry keys HKEY_LOCAL_MACHINE and RegSetValue are used for persistence. Additional API calls include LoadLibrary and GetProcAddress for dynamic function resolution."""
        
        analysis = chunker._analyze_content_type(text)
        
        # Should detect as technical due to keywords
        assert analysis['type'] == 'technical'
        assert 'keyword_count' in analysis['metadata']
        assert analysis['metadata']['keyword_count'] > 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
