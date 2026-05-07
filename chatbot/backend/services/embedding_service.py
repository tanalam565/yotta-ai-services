"""
Embedding service backed by Azure OpenAI.

Creates query/document embeddings with retry handling and shared HTTP
connection pooling.
"""

from openai import AzureOpenAI, RateLimitError, APIConnectionError
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
import config
from services.http_client_service import get_shared_http_client


class EmbeddingService:
    """Create vector embeddings for search and retrieval pipelines."""

    def __init__(self):
        """Initialize embedding client and model configuration."""
        # Use shared HTTP client for connection pooling
        self.client = AzureOpenAI(
            api_version=config.AZURE_OPENAI_EMBEDDING_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_EMBEDDING_ENDPOINT,
            api_key=config.AZURE_OPENAI_EMBEDDING_KEY,
            http_client=get_shared_http_client()  # ← SHARED POOL
        )
        self.deployment = config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        self.model = config.AZURE_OPENAI_EMBEDDING_MODEL
        self.dimensions = config.EMBEDDING_DIMENSIONS
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            "Embedding service initialized: model=%s, deployment=%s, dimensions=%s",
            self.model,
            self.deployment,
            self.dimensions,
        )

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3)
    )
    def _generate_with_retry(self, text: str) -> List[float]:
        """Execute a single embedding request with retry support from tenacity."""
        response = self.client.embeddings.create(
            input=text,
            model=self.deployment,
            dimensions=self.dimensions
        )
        return response.data[0].embedding

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text payload.

        Text is truncated to Azure token-safe limits before request execution.
        This method is synchronous; async callers should invoke via
        ``asyncio.to_thread``.

        Returns:
            List[float]: Embedding vector with configured dimensions.
        """
        try:
            if len(text) > 32000:
                text = text[:32000]

            embedding = self._generate_with_retry(text)

            if len(embedding) != self.dimensions:
                self.logger.warning("Expected %s dimensions, got %s", self.dimensions, len(embedding))

            return embedding

        except Exception as e:
            self.logger.error("Error generating embedding after retries: %s", e)
            return [0.0] * self.dimensions

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in fixed-size batches.

        Primarily used in indexing scripts where batch throughput is preferred
        over per-request latency.

        Returns:
            List[List[float]]: Embedding vectors in the same order as inputs.
        """
        all_embeddings = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                truncated_batch = [text[:32000] if len(text) > 32000 else text for text in batch]

                response = self.client.embeddings.create(
                    input=truncated_batch,
                    model=self.deployment,
                    dimensions=self.dimensions
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            self.logger.error("Error generating batch embeddings: %s", e)
            return [[0.0] * self.dimensions for _ in texts]
