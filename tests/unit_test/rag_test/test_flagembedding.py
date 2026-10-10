"""Unit tests for FlagEmbedding integration in VX-RAG.

Tests BAAI/bge-m3 dense/sparse embedding generation and
BAAI/bge-reranker-v2-m3 cross-encoder reranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.rag.libs.embeddings import BGEM3Embedding
from src.rag.libs.postprocessors import BGECrossEncoderReranker

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401


class TestBGEM3Embedding:
    """Test suite for BGEM3Embedding dense and sparse generation."""

    def test_init_parameters(self) -> None:
        """Verify initialization parameters and defaults."""
        embed_model = BGEM3Embedding(
            model_name="BAAI/bge-m3",
            threads=4,
            cache_size=100,
            embed_batch_size=16,
            device="cpu",
            normalize=True,
        )
        assert embed_model.model_name == "BAAI/bge-m3"
        assert embed_model.threads == 4
        assert embed_model.cache_size == 100
        assert embed_model.embed_dim == 1024
        assert embed_model.device == "cpu"
        assert embed_model.normalize is True

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_dense_embedding_dimension_and_caching(
        self,
        mock_model_cls: MagicMock,
    ) -> None:
        """Verify dense embeddings return 1024-dimensional vectors and utilize cache.

        Args:
            mock_model_cls: Mocked BGEM3FlagModel class.
        """
        mock_instance = MagicMock()
        mock_instance.encode.return_value = {
            "dense_vecs": [[0.1] * 1024],
        }
        mock_model_cls.return_value = mock_instance

        embed_model = BGEM3Embedding(
            model_name="BAAI/bge-m3",
            threads=2,
            cache_size=50,
            device="cpu",
        )

        text = "Test sample query for embedding"
        embedding1 = embed_model.get_query_embedding(text)
        assert len(embedding1) == 1024
        assert mock_instance.encode.call_count == 1

        # Second call must hit cache and avoid re-encoding
        embedding2 = embed_model.get_query_embedding(text)
        assert embedding2 == embedding1
        assert mock_instance.encode.call_count == 1

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_sparse_encoding_format(
        self,
        mock_model_cls: MagicMock,
    ) -> None:
        """Verify sparse vectors return integer indices and float values.

        Args:
            mock_model_cls: Mocked BGEM3FlagModel class.
        """
        mock_instance = MagicMock()
        mock_instance.encode.return_value = {
            "lexical_weights": [
                {101: 0.5, 2054: 0.8, 3000: 1.2},
                {"400": 0.3, "500": 0.7},
            ]
        }
        mock_model_cls.return_value = mock_instance

        embed_model = BGEM3Embedding(
            model_name="BAAI/bge-m3",
            device="cpu",
        )

        texts = ["First document text", "Second document text"]
        indices, values = embed_model.encode_sparse(texts)

        assert len(indices) == 2
        assert len(values) == 2
        assert indices[0] == [101, 2054, 3000]
        assert values[0] == [0.5, 0.8, 1.2]
        # String token IDs should be coerced to integers
        assert indices[1] == [400, 500]
        assert values[1] == [0.3, 0.7]

    @pytest.mark.asyncio
    async def test_async_embedding_methods(self) -> None:
        """Verify asynchronous methods return correct embeddings."""
        embed_model = BGEM3Embedding(model_name="BAAI/bge-m3", device="cpu")

        with patch.object(embed_model, "_get_text_embedding", return_value=[0.5] * 1024):
            emb = await embed_model._aget_text_embedding("Sample text")
            assert len(emb) == 1024

        with patch.object(embed_model, "_get_query_embedding", return_value=[0.2] * 1024):
            emb_q = await embed_model._aget_query_embedding("Query text")
            assert len(emb_q) == 1024

        with patch.object(
            embed_model, "_get_text_embeddings", return_value=[[0.1] * 1024, [0.2] * 1024]
        ):
            embs = await embed_model._aget_text_embeddings(["Text 1", "Text 2"])
            assert len(embs) == 2

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_encode_dense_and_sparse_single_pass(
        self,
        mock_model_cls: MagicMock,
    ) -> None:
        """Verify encode_dense_and_sparse computes both dense and sparse vectors in one pass.

        Args:
            mock_model_cls: Mocked BGEM3FlagModel class.
        """
        mock_instance = MagicMock()
        mock_instance.encode.return_value = {
            "dense_vecs": [[0.5] * 1024],
            "lexical_weights": [{101: 0.9, 202: 0.4}],
        }
        mock_model_cls.return_value = mock_instance

        embed_model = BGEM3Embedding(
            model_name="BAAI/bge-m3",
            device="cpu",
        )

        texts = ["Document for single-pass embedding"]
        dense_vecs, (indices, values) = embed_model.encode_dense_and_sparse(texts)

        assert len(dense_vecs) == 1
        assert len(dense_vecs[0]) == 1024
        assert dense_vecs[0][0] == 0.5
        assert len(indices) == 1
        assert indices[0] == [101, 202]
        assert values[0] == [0.9, 0.4]
        mock_instance.encode.assert_called_once_with(
            texts,
            batch_size=embed_model.embed_batch_size,
            max_length=embed_model.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_encode_dense_and_sparse_empty_input(
        self,
        mock_model_cls: MagicMock,
    ) -> None:
        """Verify encode_dense_and_sparse returns empty lists for empty text input."""
        embed_model = BGEM3Embedding(model_name="BAAI/bge-m3", device="cpu")
        dense_vecs, (indices, values) = embed_model.encode_dense_and_sparse([])
        assert dense_vecs == []
        assert indices == []
        assert values == []
        mock_model_cls.assert_not_called()

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_encode_dense_and_sparse_failure_raises_embedding_error(
        self,
        mock_model_cls: MagicMock,
    ) -> None:
        """Verify inference exceptions raise EmbeddingError.

        Args:
            mock_model_cls: Mocked BGEM3FlagModel class.
        """
        from src.rag.exceptions import EmbeddingError

        mock_instance = MagicMock()
        mock_instance.encode.side_effect = RuntimeError("Inference crash")
        mock_model_cls.return_value = mock_instance

        embed_model = BGEM3Embedding(model_name="BAAI/bge-m3", device="cpu")
        with pytest.raises(EmbeddingError, match="Inference crash"):
            embed_model.encode_dense_and_sparse(["Error text"])

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_encode_dense_and_sparse_whitespace_and_empty_strings(
        self,
        mock_model_cls: MagicMock,
    ) -> None:
        """Verify encode_dense_and_sparse handles empty and whitespace strings safely."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = {
            "dense_vecs": [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024],
            "lexical_weights": [{101: 0.8}, {102: 0.5}, {303: 0.9}],
        }
        mock_model_cls.return_value = mock_instance

        embed_model = BGEM3Embedding(model_name="BAAI/bge-m3", device="cpu")
        dense_vecs, (indices, values) = embed_model.encode_dense_and_sparse(
            ["", "   ", "Valid document text"]
        )

        assert len(dense_vecs) == 3
        assert indices[0] == []
        assert values[0] == []
        assert indices[1] == []
        assert values[1] == []
        assert indices[2] == [303]
        assert values[2] == [0.9]


class TestBGECrossEncoderRerankerFlagEmbedding:
    """Test suite for FlagReranker-based BGECrossEncoderReranker."""

    @patch("FlagEmbedding.FlagReranker")
    def test_reranking_with_native_sigmoid(
        self,
        mock_reranker_cls: MagicMock,
    ) -> None:
        """Verify FlagReranker scores order nodes correctly and truncate to top_n.

        Args:
            mock_reranker_cls: Mocked FlagReranker class.
        """
        mock_instance = MagicMock()
        mock_instance.compute_score.return_value = [0.15, 0.92, 0.65]
        mock_reranker_cls.return_value = mock_instance

        reranker = BGECrossEncoderReranker(
            model_name="BAAI/bge-reranker-v2-m3",
            top_n=2,
            device="cpu",
        )

        nodes = [
            NodeWithScore(node=TextNode(text="Candidate 1", id_="c1"), score=0.5),
            NodeWithScore(node=TextNode(text="Candidate 2", id_="c2"), score=0.5),
            NodeWithScore(node=TextNode(text="Candidate 3", id_="c3"), score=0.5),
        ]
        query = QueryBundle("Relevant search query")

        reranked = reranker.postprocess_nodes(nodes, query)

        assert len(reranked) == 2
        assert reranked[0].node.id_ == "c2"
        assert reranked[0].score == pytest.approx(0.92)
        assert reranked[1].node.id_ == "c3"
        assert reranked[1].score == pytest.approx(0.65)
        mock_instance.compute_score.assert_called_once()

    def test_reranker_empty_nodes_and_empty_query(self) -> None:
        """Verify handling of empty node list and empty query bundle."""
        reranker = BGECrossEncoderReranker(
            model_name="BAAI/bge-reranker-v2-m3",
            top_n=3,
        )

        # Empty nodes
        assert reranker.postprocess_nodes([], QueryBundle("query")) == []

        # Empty query
        nodes = [NodeWithScore(node=TextNode(text="Test", id_="t1"), score=0.7)]
        result = reranker.postprocess_nodes(nodes, QueryBundle(""))
        assert len(result) == 1
        assert result[0].node.id_ == "t1"
