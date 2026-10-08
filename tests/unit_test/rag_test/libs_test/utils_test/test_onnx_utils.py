"""Unit tests for ONNX runtime initialization utility."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.rag.libs.utils.onnx_utils import init_onnx_runtime
from src.utils.config_loader import get_onnx_config


class TestInitOnnxRuntime:
    """Tests for init_onnx_runtime helper."""

    def test_successful_initialization(self) -> None:
        """Test successful initialization of session and tokenizer with mocked backends."""
        mock_tokenizer = MagicMock()
        mock_session = MagicMock()
        mock_options = MagicMock()

        with (
            patch("tokenizers.Tokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("onnxruntime.SessionOptions", return_value=mock_options),
            patch("onnxruntime.InferenceSession", return_value=mock_session),
            patch("glob.glob", return_value=["/fake/path/model.onnx"]),
        ):
            session, tokenizer = init_onnx_runtime(
                model_name="BAAI/bge-m3",
                threads=8,
                component_name="embedding",
                inter_op_threads=2,
            )

            assert session is mock_session
            assert tokenizer is mock_tokenizer
            assert mock_options.intra_op_num_threads == 8
            assert mock_options.inter_op_num_threads == 2

    def test_graceful_fallback_when_models_not_cached(self) -> None:
        """Test graceful fallback returning None when no cached onnx files exist."""
        with (
            patch("tokenizers.Tokenizer.from_pretrained", side_effect=Exception("not found")),
            patch("glob.glob", return_value=[]),
        ):
            session, tokenizer = init_onnx_runtime(
                model_name="nonexistent/model",
                threads=4,
            )

            assert session is None
            assert tokenizer is None

    def test_default_inter_op_threads_used(self) -> None:
        """Test default inter-op threads parameter uses configuration value."""
        mock_options = MagicMock()

        with (
            patch("onnxruntime.SessionOptions", return_value=mock_options),
            patch("onnxruntime.InferenceSession", return_value=MagicMock()),
            patch("glob.glob", return_value=["/fake/path/model.onnx"]),
            patch("tokenizers.Tokenizer.from_pretrained", return_value=MagicMock()),
        ):
            init_onnx_runtime(
                model_name="BAAI/bge-m3",
                threads=6,
            )

            expected_threads = int(get_onnx_config()["inter_op_threads"])
            assert mock_options.inter_op_num_threads == expected_threads
