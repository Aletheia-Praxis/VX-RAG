"""
Unit tests for LLM proxy service.
"""

from unittest.mock import patch, MagicMock

from rag.services.llm_proxy.service import LLMProxy


class TestLLMProxy:
    """Test LLM proxy service."""
    
    @patch('rag.services.llm_proxy.service.Ollama')
    def test_init(self, mock_ollama: MagicMock) -> None:
        """Test initialization."""
        proxy = LLMProxy()
        mock_ollama.assert_called_once_with(model="llama3")
        assert proxy.model_name == "llama3"
    
    @patch('rag.services.llm_proxy.service.Ollama')
    def test_generate_without_context(self, mock_ollama: MagicMock) -> None:
        """Test generate without context."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.__str__ = lambda: "Generated response"
        mock_llm.complete.return_value = mock_response
        mock_ollama.return_value = mock_llm
        
        proxy = LLMProxy()
        result = proxy.generate("Test prompt")
        
        assert result == "Generated response"
        mock_llm.complete.assert_called_once_with("Test prompt")
    
    @patch('rag.services.llm_proxy.service.Ollama')
    def test_generate_with_context(self, mock_ollama: MagicMock) -> None:
        """Test generate with context."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.__str__ = lambda: "Contextual response"
        mock_llm.complete.return_value = mock_response
        mock_ollama.return_value = mock_llm
        
        proxy = LLMProxy()
        context = [{"text": "Doc 1"}, {"text": "Doc 2"}]
        result = proxy.generate("Test prompt", context)
        
        expected_prompt = "Context:\nDocument 1:\nDoc 1\n\nDocument 2:\nDoc 2\n\nQuestion: Test prompt\n\nAnswer:"
        mock_llm.complete.assert_called_once_with(expected_prompt)
        assert result == "Contextual response"
    
    @patch('rag.services.llm_proxy.service.Ollama')
    def test_generate_with_sources(self, mock_ollama: MagicMock) -> None:
        """Test generate with sources."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.__str__ = lambda: "Response with sources"
        mock_llm.complete.return_value = mock_response
        mock_ollama.return_value = mock_llm
        
        proxy = LLMProxy()
        context = [
            {"text": "Long document text here", "score": 0.9, "metadata": {"title": "Doc1"}},
            {"text": "Short", "score": 0.8, "metadata": {"title": "Doc2"}}
        ]
        result = proxy.generate_with_sources("Test query", context)
        
        assert "response" in result
        assert "sources" in result
        assert result["response"] == "Response with sources"
        assert len(result["sources"]) == 2
        assert result["sources"][0]["text"].startswith("Long document")
        assert result["sources"][0]["score"] == 0.9
