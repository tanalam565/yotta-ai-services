"""
Service class for interacting with Azure Cognitive Search.

This class provides methods to perform hybrid searches (keyword + vector) on indexed documents,
manage search indexers, and handle document retrieval with per-document chunk limiting.
It integrates with Azure Blob Storage for document downloads and uses embedding services
for vector queries.
"""

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ServiceRequestError, HttpResponseError
from services.blob_service import BlobService
from typing import List, Dict
import urllib.parse
import asyncio
import logging
import config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from services.embedding_service import EmbeddingService


class AzureSearchService:
    """Coordinate hybrid retrieval, metadata shaping, and indexer operations."""

    def __init__(self):
        """
        Initialize the Azure Search service clients and dependencies.

        Sets up:
        - Search client for document retrieval
        - Indexer client for indexing operations
        - Embedding service for vector queries
        - Blob service for download URL generation
        """
        self.endpoint = config.AZURE_SEARCH_ENDPOINT
        self.key = config.AZURE_SEARCH_KEY
        self.index_name = config.AZURE_SEARCH_INDEX_NAME
        self.indexer_name = "azureblob-indexer-yotta"

        self.credential = AzureKeyCredential(self.key)

        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )

        self.indexer_client = SearchIndexerClient(
            endpoint=self.endpoint,
            credential=self.credential
        )

        self.embedding_service = EmbeddingService()
        self.blob_service = BlobService()
        self.logger = logging.getLogger(__name__)

        self.logger.info("Connected to index: %s (Hybrid Search enabled)", self.index_name)
        self.logger.info("Max chunks per document: %s", config.MAX_CHUNKS_PER_DOCUMENT)

    def _extract_filename(self, result_dict: dict) -> str:
        """
        Extract a human-readable filename from a search result payload.

        Attempts title, filepath, and parent_id path extraction in order.

        Returns:
            str: Best-effort filename, or ``"Unknown Document"`` when missing.
        """
        title = result_dict.get("title")
        if title and title.strip():
            return title

        filepath = result_dict.get("filepath")
        if filepath and filepath.strip():
            return filepath.split("/")[-1] if "/" in filepath else filepath

        parent_id = result_dict.get("parent_id")
        if parent_id and parent_id.strip():
            try:
                parsed = urllib.parse.urlparse(parent_id)
                path = parsed.path
                if '/' in path:
                    filename = path.split('/')[-1]
                    filename = urllib.parse.unquote(filename)
                    if filename:
                        return filename
            except:
                pass

        return "Unknown Document"

    # ── Sync helpers (run via asyncio.to_thread) ─────────────────────────────────

    @retry(
        retry=retry_if_exception_type((ServiceRequestError, HttpResponseError)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3)
    )
    def _execute_search_sync(self, query: str, vector_query, top: int) -> list:
        """Execute synchronous hybrid search and return materialized result dictionaries."""
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top * 5,
            include_total_count=True
        )
        return [dict(r) for r in results]

    @retry(
        retry=retry_if_exception_type((ServiceRequestError, HttpResponseError)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3)
    )
    def _execute_keyword_search_sync(self, query: str, top: int) -> list:
        """Execute synchronous keyword-only search and return result dictionaries."""
        results = self.search_client.search(
            search_text=query,
            top=top * 3,
            include_total_count=True
        )
        return [dict(r) for r in results]

    def _get_indexer_status_sync(self):
        """Fetch current indexer status synchronously via Azure SDK client."""
        return self.indexer_client.get_indexer_status(self.indexer_name)

    def _run_indexer_sync(self):
        """Trigger Azure Search indexer run synchronously."""
        self.indexer_client.run_indexer(self.indexer_name)

    # ── Async public methods ──────────────────────────────────────────────────────

    async def search(self, query: str, top: int = config.MAX_SEARCH_RESULTS) -> List[Dict]:
        """
        Perform hybrid search (keyword + vector) with per-document chunk limiting.

        Workflow:
        1. Generate query embedding
        2. Execute hybrid Azure Search query
        3. Limit chunks per parent document
        4. Attach source metadata/download URLs

        Args:
            query: User query text.
            top: Maximum number of chunks to return.

        Returns:
            List[Dict]: Ranked chunk payloads for LLM context.
        """
        try:
            self.logger.info("Hybrid search for query='%s' target_results=%s", query, top)

            # Generate query embedding off the event loop
            query_embedding = await asyncio.to_thread(
                self.embedding_service.generate_embedding, query
            )

            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top * 8,
                fields="content_vector"
            )

            # Execute search off the event loop
            raw_results = await asyncio.to_thread(
                self._execute_search_sync, query, vector_query, top
            )

            # Group chunks by parent_id and limit per document
            parent_chunks = {}
            processed_results = []

            for result_dict in raw_results:
                parent_id = result_dict.get("parent_id")
                if not parent_id:
                    parent_id = result_dict.get("chunk_id", f"standalone_{len(parent_chunks)}")

                if parent_id not in parent_chunks:
                    parent_chunks[parent_id] = {
                        'count': 0,
                        'chunks': [],
                        'filename': self._extract_filename(result_dict)
                    }

                if parent_chunks[parent_id]['count'] >= config.MAX_CHUNKS_PER_DOCUMENT:
                    continue

                content = result_dict.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(item) for item in content)

                if not content:
                    continue

                filename = parent_chunks[parent_id]['filename']

                blob_name = result_dict.get("metadata_storage_name", "")
                download_url = None
                if blob_name:
                    try:
                        download_url = self.blob_service.generate_download_url(blob_name)
                    except Exception as e:
                        self.logger.warning("Error generating download URL for %s: %s", blob_name, e)

                chunk_data = {
                    "content": str(content)[:5000],
                    "filename": filename,
                    "source_type": "company",
                    "download_url": download_url,
                    "parent_id": parent_id,
                    "chunk_number": result_dict.get("chunk_number"),
                    "page_number": result_dict.get("page_number", 1)
                }

                parent_chunks[parent_id]['chunks'].append(chunk_data)
                parent_chunks[parent_id]['count'] += 1
                processed_results.append(chunk_data)

                if len(processed_results) >= top:
                    break

            self.logger.info(
                "Retrieval stats: unique_documents=%s, chunks_retrieved=%s",
                len(parent_chunks),
                len(processed_results),
            )

            return processed_results[:top]

        except Exception as e:
            self.logger.exception("Hybrid search error: %s", e)
            return await self._fallback_keyword_search(query, top)

    async def _fallback_keyword_search(self, query: str, top: int) -> List[Dict]:
        """
        Fallback keyword-only retrieval path used when hybrid search fails.

        Returns:
            List[Dict]: Best-effort chunk list without vector ranking.
        """
        try:
            self.logger.warning("Falling back to keyword-only search")

            raw_results = await asyncio.to_thread(
                self._execute_keyword_search_sync, query, top
            )

            parent_chunks = {}
            search_results = []

            for result_dict in raw_results:
                parent_id = result_dict.get("parent_id")
                if not parent_id:
                    parent_id = result_dict.get("chunk_id", f"standalone_{len(parent_chunks)}")

                if parent_id not in parent_chunks:
                    parent_chunks[parent_id] = 0

                if parent_chunks[parent_id] >= config.MAX_CHUNKS_PER_DOCUMENT:
                    continue

                content = result_dict.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(item) for item in content)

                filename = self._extract_filename(result_dict)

                if filename == "Unknown Document":
                    continue

                blob_name = result_dict.get("metadata_storage_name", "")
                download_url = None
                if blob_name:
                    download_url = self.blob_service.generate_download_url(blob_name)

                if content:
                    search_results.append({
                        "content": str(content)[:5000],
                        "filename": filename,
                        "source_type": "company",
                        "download_url": download_url,
                        "page_number": result_dict.get("page_number", 1)
                    })
                    parent_chunks[parent_id] += 1

                if len(search_results) >= top:
                    break

            self.logger.info(
                "Keyword fallback returned %s results from %s documents",
                len(search_results),
                len(parent_chunks),
            )
            return search_results

        except Exception as e:
            self.logger.exception("Fallback search error: %s", e)
            return []

    async def get_indexer_status(self):
        """Get current Azure Search indexer status and latest execution result details."""
        try:
            status = await asyncio.to_thread(self._get_indexer_status_sync)
            return {
                "name": status.name,
                "status": status.status,
                "last_result": {
                    "status": status.last_result.status if status.last_result else None,
                    "error_message": status.last_result.error_message if status.last_result else None
                }
            }
        except Exception as e:
            self.logger.error("Error getting indexer status: %s", e)
            return {"error": str(e)}

    async def run_indexer(self):
        """Manually trigger the configured Azure Search indexer run."""
        try:
            await asyncio.to_thread(self._run_indexer_sync)
            self.logger.info("Indexer '%s' triggered successfully", self.indexer_name)
            return True
        except Exception as e:
            self.logger.error("Error running indexer: %s", e)
            return False