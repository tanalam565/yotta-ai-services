"""
Central configuration for the YottaReal Bot API.

This module is the single source of truth for runtime settings loaded from
environment variables. Values are grouped by concern so all services use
consistent settings for:

- Azure Blob Storage and Search
- Azure OpenAI chat and embedding deployments
- Azure Document Intelligence extraction
- Redis sessions/history
- Upload limits, rate limiting, and request timeouts
- CORS behavior for frontend access

Most values include development-friendly defaults. Production deployments
should provide explicit environment variables for all sensitive settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Azure Blob Storage Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "filescontainer")
AZURE_UPLOADS_CONTAINER_NAME = os.getenv("AZURE_UPLOADS_CONTAINER_NAME", "user-uploads")

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "azureblob-index-yotta")
AZURE_SEARCH_DATASOURCE_NAME = os.getenv("AZURE_SEARCH_DATASOURCE_NAME", "property-blob-datasource")
AZURE_SEARCH_INDEXER_NAME = os.getenv("AZURE_SEARCH_INDEXER_NAME", "azureblob-indexer-yotta")

# Azure OpenAI Configuration (for chat)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-chat")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

# Azure OpenAI Embeddings Configuration (for hybrid search)
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "https://yotta-openai-service.openai.azure.com/")
AZURE_OPENAI_EMBEDDING_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_KEY", "")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-12-01-preview")
EMBEDDING_DIMENSIONS = 3072

# Azure Document Intelligence Configuration
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")

# API Key Authentication
CHATBOT_API_KEY = os.getenv("CHATBOT_API_KEY", "")

# Base directory used for local durable storage paths
_BASE_DIR = os.path.dirname(__file__)

# Application Settings
MAX_SEARCH_RESULTS = 15
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Settings
MAX_CHUNKS_PER_DOCUMENT = 7

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Session and History Settings
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "7200"))  # 2 hours
MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "10"))

# File Upload Limits
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "15"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_UPLOAD_PAGES = int(os.getenv("MAX_UPLOAD_PAGES", "15"))
MAX_UPLOADS_PER_SESSION = int(os.getenv("MAX_UPLOADS_PER_SESSION", "5"))

# Rate Limiting
RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "20/minute")
RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "5/minute")

# Request Timeouts
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "90"))

# Durable storage (source of truth)
DATABASE_URL = os.getenv("DATABASE_URL", "")
PERSISTENCE_DB_PATH = os.getenv(
	"PERSISTENCE_DB_PATH",
	os.path.join(_BASE_DIR, "data", "chat_logs.db")
)

# CORS - comma-separated list of allowed origins
def _parse_cors_allowed_origins(raw_value: str) -> list[str]:
	"""Parse comma-separated CORS origins into a normalized list."""
	if not raw_value:
		return []
	origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
	if "*" in origins and len(origins) > 1:
		raise ValueError("CORS_ALLOWED_ORIGINS cannot mix '*' with specific origins")
	return origins


CORS_ALLOWED_ORIGINS = _parse_cors_allowed_origins(
	os.getenv("CORS_ALLOWED_ORIGINS", "")
)