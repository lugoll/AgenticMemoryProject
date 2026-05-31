"""Shim module loaded by the LiteLLM proxy at startup.

The proxy's ``get_instance_fn`` resolves ``custom_handler`` references in
``proxy_config.yaml`` by file path relative to the YAML location, not by
regular Python import. So the handler instance has to live next to the
config; this shim simply re-exports the real implementation from
``src/utils/local_embedding_provider.py``.
"""
from src.utils.local_embedding_provider import local_bge_handler

__all__ = ["local_bge_handler"]
