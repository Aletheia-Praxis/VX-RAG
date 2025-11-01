"""
Tests for MCP bridge module.
"""

import unittest
from unittest.mock import Mock, patch
from src.mcp.bridge import redact_sensitive_data, MCPBridge

class TestBridge(unittest.TestCase):
    """
    Test cases for MCP bridge.
    """

    def test_redact_sensitive_data(self) -> None:
        """
        Test redaction of sensitive data.
        """
        # Test email redaction
        text_with_email = "Contact me at user@example.com for details."
        redacted = redact_sensitive_data(text_with_email)
        self.assertIn("[REDACTED_EMAIL]", redacted)
        self.assertNotIn("user@example.com", redacted)
        
        # Test IP redaction
        text_with_ip = "Server IP is 192.168.1.1"
        redacted = redact_sensitive_data(text_with_ip)
        self.assertIn("[REDACTED_IP]", redacted)
        self.assertNotIn("192.168.1.1", redacted)
        
        # Test multiple redactions
        text_mixed = "Email: test@domain.com and IP: 10.0.0.1"
        redacted = redact_sensitive_data(text_mixed)
        self.assertIn("[REDACTED_EMAIL]", redacted)
        self.assertIn("[REDACTED_IP]", redacted)
        self.assertNotIn("test@domain.com", redacted)
        self.assertNotIn("10.0.0.1", redacted)
        
        # Test no sensitive data
        text_clean = "This is a normal text without sensitive data."
        redacted = redact_sensitive_data(text_clean)
        self.assertEqual(redacted, text_clean)

    def test_handle_query(self) -> None:
        """
        Test handling queries.
        """
        # TODO: Implement test
        # mock_engine = Mock()
        # bridge = MCPBridge(mock_engine)
        # response = bridge.handle_query("test query")
        # self.assertIn("query", response)
        pass

    def test_query_documents_with_redaction(self) -> None:
        """
        Test query_documents with redaction and citations.
        """
        # Mock retriever
        mock_retriever = Mock()
        mock_docs = [
            {
                'text': 'Contact at user@example.com for info. Server IP: 192.168.1.1',
                'score': 0.9,
                'metadata': {
                    'file_name': 'document.pdf',
                    'title': 'Test Document',
                    'file_type': 'pdf',
                    'page_count': 10
                },
                'node_id': 'test_1'
            }
        ]
        mock_retriever.retrieve.return_value = mock_docs
        
        # Create bridge with mocked retriever
        bridge = MCPBridge()
        bridge.retriever = mock_retriever
        
        # Test query
        result = bridge.query_documents("test query", top_k=1)
        
        # Check structure
        self.assertIn("query", result)
        self.assertIn("response", result)
        self.assertIn("sources", result)
        
        # Check redaction in response
        self.assertIn("[REDACTED_EMAIL]", result["response"])
        self.assertIn("[REDACTED_IP]", result["response"])
        self.assertNotIn("user@example.com", result["response"])
        self.assertNotIn("192.168.1.1", result["response"])
        
        # Check redaction in sources
        source = result["sources"][0]
        self.assertIn("[REDACTED_EMAIL]", source["text"])
        self.assertIn("[REDACTED_IP]", source["text"])
        self.assertNotIn("user@example.com", source["text"])
        self.assertNotIn("192.168.1.1", source["text"])
        
        # Check citations (metadata)
        self.assertEqual(source["metadata"]["file_name"], "document.pdf")
        self.assertEqual(source["metadata"]["title"], "Test Document")
        self.assertEqual(source["metadata"]["page_count"], 10)

if __name__ == "__main__":
    unittest.main()
