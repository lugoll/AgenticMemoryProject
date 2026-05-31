"""
LiteLLM CustomLLM that serves the project's existing
``BAAI/bge-base-en-v1.5`` sentence-transformer over an OpenAI-style
embedding interface.

Routes MS GraphRAG's embedding requests through the same encoder the
'vector' variant uses, so the two variants are directly comparable on
retrieval quality (the chat LLM is the only confounder).

Wired up in src/memory/_graphrag_templates/proxy_config.yaml via:

    model_list:
      - model_name: local-bge
        litellm_params:
          model: local-bge/sentence-transformers
          custom_llm_provider: local_bge
"""
from __future__ import annotations

from typing import Any

from litellm import CustomLLM
from litellm.types.utils import EmbeddingResponse, Usage


# Heavy import — loading the model takes a few seconds. We pay it once per
# proxy worker, then reuse across requests.
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        # Delay sentence-transformers import to keep cold-start fast for
        # variants that don't need it.
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    return _encoder


def _encode(texts: list[str]) -> tuple[list[list[float]], int]:
    """Return (embeddings, approx_token_count). Tokens are approximated by the
    encoder's tokenizer so the proxy can report usage consistently with the
    sentence-transformer the 'vector' variant already uses."""
    encoder = _get_encoder()
    vectors = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    # bge-base produces dense float32 vectors; LiteLLM expects lists.
    embeddings = vectors.tolist()
    # Token count via the encoder's own tokenizer.
    tokenizer = encoder.tokenizer
    n_tokens = sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)
    return embeddings, n_tokens


def _build_response(
    model: str,
    embeddings: list[list[float]],
    n_tokens: int,
    model_response: EmbeddingResponse,
) -> EmbeddingResponse:
    model_response.model = model
    model_response.object = "list"
    model_response.data = [
        {"object": "embedding", "index": i, "embedding": emb}
        for i, emb in enumerate(embeddings)
    ]
    # Embedding responses have no completion tokens; total == prompt.
    model_response.usage = Usage(
        prompt_tokens=n_tokens,
        completion_tokens=0,
        total_tokens=n_tokens,
    )
    return model_response


class LocalBGEEmbedding(CustomLLM):
    """OpenAI-compatible embedding provider backed by sentence-transformers."""

    def embedding(  # type: ignore[override]
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Any,
        logging_obj: Any,
        optional_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: Any = None,
        litellm_params: Any = None,
    ) -> EmbeddingResponse:
        # OpenAI clients sometimes send a single string instead of a list.
        texts = input if isinstance(input, list) else [input]
        embeddings, n_tokens = _encode([str(t) for t in texts])
        return _build_response(model, embeddings, n_tokens, model_response)

    async def aembedding(  # type: ignore[override]
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Any,
        logging_obj: Any,
        optional_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: Any = None,
        litellm_params: Any = None,
    ) -> EmbeddingResponse:
        # sentence-transformers is sync; running it in the event loop is fine
        # because the proxy gates concurrency via concurrent_requests=1.
        texts = input if isinstance(input, list) else [input]
        embeddings, n_tokens = _encode([str(t) for t in texts])
        return _build_response(model, embeddings, n_tokens, model_response)


# Singleton handler — referenced by name from proxy_config.yaml via
# ``custom_llm_provider: src.utils.local_embedding_provider.local_bge_handler``.
local_bge_handler = LocalBGEEmbedding()
