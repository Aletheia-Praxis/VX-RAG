"""
Configuration module for VX-RAG.

Loads settings from YAML files and environment variables.
"""

import os
from pathlib import Path
import yaml

class Config:
    """
    Configuration class for the RAG system.
    """

    def __init__(self, config_file: str = "../config/settings.yaml"):
        """
        Load configuration from file and environment.

        Args:
            config_file: Path to the YAML config file.
        """
        self.config = self._load_config(config_file)
        self.data_dir = Path(self.config.get("data_dir", "../data"))
        self.index_dir = Path(self.config.get("index_dir", "../data/index"))
        self.embedding_model = self.config.get("embedding_model", "...")
        self.chunk_size = self.config.get("chunk_size", 512)
        self.llm_api_key = os.getenv("LLM_API_KEY")

    def _load_config(self, config_file: str) -> dict:
        """
        Load YAML configuration file.

        Args:
            config_file: Path to config file.

        Returns:
            Dictionary of config values.
        """
        if Path(config_file).exists():
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        return {}

# Global config instance
config = Config()