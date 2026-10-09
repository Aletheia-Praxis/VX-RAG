"""Unit tests for ONNX runtime initialization and path resolution utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import onnxruntime
import pytest

from src.rag.libs.utils.onnx_utils import (
    create_onnx_session_options,
    get_fastembed_cross_encoder,
    get_fastembed_text_embedding,
    init_onnx_runtime,
    resolve_onnx_model_path,
)


class TestCreateOnnxSessionOptions:
    """Tests for create_onnx_session_options."""

    def test_sequential_mode_when_inter_op_threads_one(self) -> None:
        """Verify ORT_SEQUENTIAL mode is set without configuring inter_op_num_threads."""
        options = create_onnx_session_options(threads=4, inter_op_threads=1)
        assert options.intra_op_num_threads == 4
        assert options.execution_mode == onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    def test_parallel_mode_when_inter_op_threads_greater_than_one(self) -> None:
        """Verify ORT_PARALLEL mode is configured with inter_op_num_threads."""
        options = create_onnx_session_options(threads=4, inter_op_threads=4)
        assert options.intra_op_num_threads == 4
        assert options.execution_mode == onnxruntime.ExecutionMode.ORT_PARALLEL
        assert options.inter_op_num_threads == 4

    def test_explicit_parallel_execution_mode(self) -> None:
        """Verify explicit ORT_PARALLEL mode configures inter-op threads."""
        options = create_onnx_session_options(
            threads=2,
            inter_op_threads=3,
            execution_mode=onnxruntime.ExecutionMode.ORT_PARALLEL,
        )
        assert options.intra_op_num_threads == 2
        assert options.execution_mode == onnxruntime.ExecutionMode.ORT_PARALLEL
        assert options.inter_op_num_threads == 3


class TestResolveOnnxModelPath:
    """Tests for resolve_onnx_model_path."""

    def test_resolve_direct_existing_file(self, tmp_path: Path) -> None:
        """Verify resolving a direct file path returns the path."""
        fake_model = tmp_path / "model.onnx"
        fake_model.write_text("dummy", encoding="utf-8")

        result = resolve_onnx_model_path(str(fake_model))
        assert result == fake_model

    def test_resolve_directory_with_candidate_file(self, tmp_path: Path) -> None:
        """Verify resolving a directory containing model.onnx."""
        fake_model = tmp_path / "model.onnx"
        fake_model.write_text("dummy", encoding="utf-8")

        result = resolve_onnx_model_path(str(tmp_path))
        assert result == fake_model

    def test_resolve_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError is raised if model cannot be located."""
        with pytest.raises(FileNotFoundError, match="Could not locate ONNX model file"):
            resolve_onnx_model_path(str(tmp_path / "nonexistent_model"))


class TestFastEmbedHelpers:
    """Tests for official FastEmbed helper functions."""

    @patch("fastembed.TextEmbedding")
    def test_get_fastembed_text_embedding(self, mock_cls: MagicMock) -> None:
        """Verify get_fastembed_text_embedding instantiates TextEmbedding correctly."""
        get_fastembed_text_embedding("BAAI/bge-m3", threads=4)
        mock_cls.assert_called_once_with(
            model_name="BAAI/bge-m3",
            threads=4,
        )

    @patch("fastembed.rerank.cross_encoder.TextCrossEncoder")
    def test_get_fastembed_cross_encoder(self, mock_cls: MagicMock) -> None:
        """Verify get_fastembed_cross_encoder instantiates TextCrossEncoder correctly."""
        get_fastembed_cross_encoder("BAAI/bge-reranker-v2-m3", threads=2)
        mock_cls.assert_called_once_with(
            model_name="BAAI/bge-reranker-v2-m3",
            threads=2,
        )


class TestInitOnnxRuntime:
    """Tests for init_onnx_runtime helper."""

    def test_successful_initialization(self, tmp_path: Path) -> None:
        """Test successful initialization of session and tokenizer with mocked backends."""
        fake_model = tmp_path / "model.onnx"
        fake_model.write_text("dummy", encoding="utf-8")

        mock_tokenizer = MagicMock()
        mock_session = MagicMock()
        mock_options = MagicMock()

        with (
            patch("tokenizers.Tokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("src.rag.libs.utils.onnx_utils.resolve_onnx_model_path", return_value=fake_model),
            patch("src.rag.libs.utils.onnx_utils.create_onnx_session_options", return_value=mock_options),
            patch("onnxruntime.InferenceSession", return_value=mock_session),
        ):
            session, tokenizer = init_onnx_runtime(
                model_name="BAAI/bge-m3",
                threads=8,
                component_name="embedding",
                inter_op_threads=2,
            )

            assert session is mock_session
            assert tokenizer is mock_tokenizer

    def test_raises_runtime_error_when_tokenizer_fails(self) -> None:
        """Test descriptive RuntimeError when tokenizer initialization fails."""
        with (
            patch("tokenizers.Tokenizer.from_pretrained", side_effect=Exception("not found")),
            pytest.raises(RuntimeError, match="Failed to initialize tokenizer for model"),
        ):
            init_onnx_runtime(
                model_name="nonexistent/model",
                threads=4,
            )

    def test_raises_file_not_found_when_model_missing(self) -> None:
        """Test FileNotFoundError when ONNX model cannot be resolved."""
        mock_tokenizer = MagicMock()
        with (
            patch("tokenizers.Tokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("src.rag.libs.utils.onnx_utils.resolve_onnx_model_path", side_effect=FileNotFoundError("missing")),
            pytest.raises(FileNotFoundError, match="missing"),
        ):
            init_onnx_runtime(
                model_name="BAAI/bge-m3",
                threads=4,
            )

    def test_default_inter_op_threads_used(self, tmp_path: Path) -> None:
        """Test default inter-op threads parameter uses configuration value."""
        fake_model = tmp_path / "model.onnx"
        fake_model.write_text("dummy", encoding="utf-8")

        with (
            patch("onnxruntime.InferenceSession", return_value=MagicMock()),
            patch("src.rag.libs.utils.onnx_utils.resolve_onnx_model_path", return_value=fake_model),
            patch("tokenizers.Tokenizer.from_pretrained", return_value=MagicMock()),
            patch("src.rag.libs.utils.onnx_utils.create_onnx_session_options") as mock_create_options,
        ):
            init_onnx_runtime(
                model_name="BAAI/bge-m3",
                threads=6,
            )

            mock_create_options.assert_called_once_with(
                threads=6,
                inter_op_threads=0,
                execution_mode=None,
            )
