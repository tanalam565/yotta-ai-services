"""
YottaReal Bot API - FastAPI Backend.

This module defines the public API for chat, upload, cleanup, and operational
endpoints. It wires request validation, rate limiting, authentication, and
service orchestration for Azure Search, Azure OpenAI, and Document Intelligence.

This module provides the main FastAPI application for the YottaReal chatbot API.
It handles chat interactions, document uploads, session management, and integrates
with Azure Cognitive Search, Azure OpenAI, and Azure Document Intelligence services.

Key Features:
- Chat endpoint with hybrid search (vector + keyword) and LLM response generation
- Document upload with automatic text extraction using Azure Document Intelligence
- Session-based document management for temporary user uploads
- API key authentication
- Indexer status and manual trigger endpoints
- Health check endpoint

Endpoints:
- POST /api/chat: Process chat messages with document context
- POST /api/upload: Upload documents for session-based queries
- POST /api/cleanup-session: Clean up session documents
- GET /api/indexer/status: Get Azure Search indexer status
- POST /api/indexer/run: Manually trigger indexer
- GET /api/health: Health check (public)

Environment Variables:
- CHATBOT_API_KEY: API key for authentication
- Various Azure service configurations (see config.py)

Dependencies:
- fastapi: Web framework
- uvicorn: ASGI server
- Azure Cognitive Search, OpenAI, Document Intelligence services
"""
from contextlib import asynccontextmanager
import logging
import asyncio
from fastapi import FastAPI, HTTPException, Security, Depends, UploadFile, File, Form, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import uuid
import json

from services.azure_search_service import AzureSearchService
from services.llm_service import LLMService
from services.document_intelligence_service import DocumentIntelligenceService
from services.redis_service import get_redis_client, close_redis
from services.blob_service import BlobService
from services.chat_storage_service import PersistenceService
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if not config.CHATBOT_API_KEY:
    logger.warning("CHATBOT_API_KEY is not configured; admin-only endpoints requiring API key will be unavailable.")

# Reduce noisy third-party logs while keeping app/service logs readable.
for noisy_logger in [
    "httpx",
    "httpcore",
    "azure",
    "azure.core.pipeline.policies.http_logging_policy",
    "urllib3",
    "openai",
    "gunicorn.access",
    "uvicorn.access",
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# ── File validation via magic bytes (not trusting content-type header) ──────────
ALLOWED_SIGNATURES = [
    b'%PDF',              # PDF
    b'\xff\xd8\xff',      # JPEG
    b'\x89PNG\r\n\x1a\n', # PNG
    b'II*\x00',           # TIFF little-endian
    b'MM\x00*',           # TIFF big-endian
    b'BM',                # BMP
    b'PK\x03\x04',        # DOCX (ZIP-based)
]

ALLOWED_CONTENT_TYPES = [
    'application/pdf',
    'image/jpeg',
    'image/jpg',
    'image/png',
    'image/tiff',
    'image/bmp',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain'
]


def validate_file_content(content: bytes, content_type: str) -> bool:
    """
    Validate uploaded file bytes against expected signatures.

    Uses magic-byte checks for binary types and UTF-8 probe logic for plain text.

    Args:
        content: Raw file bytes.
        content_type: Declared MIME type from upload metadata.

    Returns:
        bool: True when content appears to match supported file type rules.
    """
    for sig in ALLOWED_SIGNATURES:
        if content[:len(sig)] == sig:
            return True
    # Plain text has no reliable magic bytes — attempt UTF-8 decode
    if content_type == 'text/plain':
        try:
            content[:1024].decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False
    return False


# Lifespan: close Redis pool on shutdown to prevent resource leaks 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup/shutdown resource management."""
    await persistence_service.initialize()
    yield
    await close_redis()


RATE_WINDOW_SECONDS = {
    "second": 1,
    "seconds": 1,
    "minute": 60,
    "minutes": 60,
    "hour": 3600,
    "hours": 3600,
    "day": 86400,
    "days": 86400,
}


def parse_rate_limit(rate_limit: str) -> tuple[int, int]:
    """Parse limits like '20/minute' into (max_requests, window_seconds)."""
    try:
        amount_raw, window_raw = rate_limit.strip().split("/", 1)
        max_requests = int(amount_raw)
        window_seconds = RATE_WINDOW_SECONDS.get(window_raw.strip().lower())
        if max_requests <= 0 or not window_seconds:
            raise ValueError("Invalid rate limit format")
        return max_requests, window_seconds
    except Exception as e:
        logger.error("Invalid rate limit config '%s': %s", rate_limit, e)
        raise RuntimeError(f"Invalid rate limit config: {rate_limit}")


async def enforce_session_rate_limit(session_id: str, action: str, rate_limit: str):
    """Enforce per-session rate limit using Redis counters."""
    max_requests, window_seconds = parse_rate_limit(rate_limit)
    redis_client = await get_redis_client()
    key = f"ratelimit:{action}:{session_id}"

    current_count = await redis_client.incr(key)
    if current_count == 1:
        await redis_client.expire(key, window_seconds)

    if current_count > max_requests:
        retry_after = await redis_client.ttl(key)
        raise HTTPException(
            status_code=429,
            detail={
                "message": f"Rate limit exceeded for session '{session_id}'",
                "limit": rate_limit,
                "retry_after_seconds": max(retry_after, 0),
            },
        )

app = FastAPI(title="Property Management Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


search_service = AzureSearchService()
llm_service = LLMService()
doc_intelligence_service = DocumentIntelligenceService()
blob_service = BlobService()
persistence_service = PersistenceService()

# API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify API key for authentication.

    Args:
        api_key: API key provided in `X-API-Key` header.

    Returns:
        bool: True if authentication succeeds.

    Raises:
        HTTPException: If key is missing/invalid when auth is enabled.
    """
    if api_key != config.CHATBOT_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return True


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    sources: List[dict]
    session_id: str


class CleanupRequest(BaseModel):
    """Request model for session cleanup endpoint."""
    session_id: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest, authenticated: bool = Depends(verify_api_key)):
    """
    Process chat messages with session uploads and indexed document retrieval.

    Args:
        request: FastAPI request (used by limiter middleware).
        body: Chat request payload.
        authenticated: Authentication dependency guard.

    Returns:
        ChatResponse: AI response with source citations and session id.
    """
    try:
        if not body.session_id:
            body.session_id = str(uuid.uuid4())

        await enforce_session_rate_limit(
            session_id=body.session_id,
            action="chat",
            rate_limit=config.RATE_LIMIT_CHAT,
        )

        logger.info(f"Chat request - Session ID: {body.session_id}, Query: {body.message}")

        # GET ALL UPLOADED DOCUMENTS FOR THIS SESSION (REDIS) 
        session_context = []
        redis_client = await get_redis_client()

        if body.session_id:
            session_key = f"session:{body.session_id}"
            session_data = await redis_client.get(session_key)
            session_docs = json.loads(session_data) if session_data else []

            # Refresh TTL on access
            if session_data:
                await redis_client.expire(session_key, config.SESSION_TTL_SECONDS)

            for doc in session_docs:
                page_entries = doc.get('page_texts') or []
                non_empty_pages = [
                    page_info for page_info in page_entries
                    if (page_info.get('text') or '').strip()
                ]

                if non_empty_pages:
                    for page_info in non_empty_pages:
                        session_context.append({
                            "content": page_info['text'],
                            "filename": doc["filename"],
                            "source_type": "uploaded",
                            "page_number": page_info['page_number']
                        })
                else:
                    fallback_content = (doc.get("content") or "").strip()
                    if not fallback_content:
                        continue
                    session_context.append({
                        "content": fallback_content,
                        "filename": doc["filename"],
                        "source_type": "uploaded",
                        "page_number": 1
                    })

            logger.info(f"Uploaded documents in session: {len(session_docs)} files")
        else:
            logger.info("No uploaded documents in this session")

        # CHECK IF CASUAL CHAT
        casual_patterns = [
            'hi', 'hello', 'hey', 'how are you', 'thanks',
            'thank you', 'bye', 'goodbye', 'good morning', 'good evening',
            'sup', 'what\'s up', 'wassup', 'yo', 'howdy', 'good night'
        ]

        query_lower = body.message.lower().strip()
        is_casual = False

        if query_lower in casual_patterns:
            is_casual = True
        elif len(query_lower.split()) <= 2:
            if any(p in query_lower for p in casual_patterns):
                is_casual = True
        elif any(pattern in query_lower for pattern in ['how are', 'how r u', 'how r you', 'hows it going', 'how do you do']):
            is_casual = True
        elif len(query_lower.split()) == 1 and len(query_lower) <= 6:
            is_casual = True

        logger.info(f"Query type: {'Casual chat' if is_casual else 'Document query'}")

        # SEARCH COMPANY DOCUMENTS
        indexed_results = []
        if not is_casual:
            logger.info("Searching company documents")
            indexed_results = await search_service.search(body.message)
            for doc in indexed_results:
                doc["source_type"] = "company"
            logger.info(f"Found {len(indexed_results)} company documents")
        else:
            logger.info("Skipping document search (casual chat)")

        # BUILD CONTEXT FOR LLM
        all_context = []

        if is_casual:
            all_context = []
            logger.info("Context for LLM: Empty (casual chat)")
        elif session_context:
            all_context = session_context + indexed_results[:15]
            logger.info(f"Context for LLM: {len(all_context)} document pages")
        else:
            all_context = indexed_results[:15]
            logger.info(f"Context for LLM: {len(all_context)} company documents")

        # LOG WHAT'S BEING SENT
        logger.info(f"Sending to LLM ({len(all_context)} document pages)")

        if not all_context and not is_casual:
            logger.warning("No documents in context for non-casual query")

        # GENERATE RESPONSE
        response = await llm_service.generate_response(
            query=body.message,
            context=all_context,
            session_id=body.session_id,
            has_uploads=bool(session_context),
            is_comparison=False
        )

        # Deduplicate sources by filename
        source_map = {}
        for source in response["sources"]:
            filename = source.get("filename", "Unknown")
            if filename not in source_map:
                source_map[filename] = source

        unique_sources = list(source_map.values())

        logger.info(f"Sources after deduplication: {len(unique_sources)}")

        try:
            await persistence_service.save_chat_exchange(
                session_id=response["session_id"],
                query=body.message,
                answer=response["answer"],
                sources=unique_sources,
            )
        except Exception as persistence_error:
            logger.warning("Failed to persist chat exchange: %s", persistence_error)

        return ChatResponse(
            response=response["answer"],
            sources=unique_sources,
            session_id=response["session_id"]
        )

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing chat request")


@app.post("/api/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    authenticated: bool = Depends(verify_api_key)
):
    """
    Upload a document, extract text, and store results in Redis session state.

    Applies content-type, signature, page, size, and per-session upload limits.
    """
    try:
        if not session_id:
            session_id = str(uuid.uuid4())

        await enforce_session_rate_limit(
            session_id=session_id,
            action="upload",
            rate_limit=config.RATE_LIMIT_UPLOAD,
        )

        logger.info(f"Upload request - Session ID: {session_id}, Filename: {file.filename}, Content-Type: {file.content_type}")

        await persistence_service.ensure_session(session_id)

        # Check upload count for this session
        redis_client = await get_redis_client()
        session_key = f"session:{session_id}"
        session_data = await redis_client.get(session_key)
        current_docs = json.loads(session_data) if session_data else []

        if len(current_docs) >= config.MAX_UPLOADS_PER_SESSION:
            logger.warning(f"Upload limit reached: {len(current_docs)}/{config.MAX_UPLOADS_PER_SESSION}")
            raise HTTPException(
                status_code=400,
                detail=f"Upload limit reached. Maximum {config.MAX_UPLOADS_PER_SESSION} files per session."
            )

        # Validate content-type header
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file.content_type} not supported"
            )

        # Read file content
        file_content = await file.read()
        logger.info(f"File size: {len(file_content)} bytes")

        # Validate file size
        if len(file_content) > config.MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds {config.MAX_FILE_SIZE_MB}MB limit"
            )

        # Validate file content via magic bytes
        if not validate_file_content(file_content, file.content_type):
            raise HTTPException(
                status_code=400,
                detail="File content does not match its declared type"
            )

        # Extract text using Document Intelligence
        logger.info(f"Extracting text from {file.filename}")
        extraction_result = await doc_intelligence_service.extract_text(
            file_content,
            file.filename
        )

        if not extraction_result['success']:
            logger.error(f"Extraction failed: {extraction_result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process uploaded file"
            )

        logger.info(f"Extracted {len(extraction_result['text'])} characters from {extraction_result['page_count']} pages")

        blob_info = await asyncio.to_thread(
            blob_service.upload_user_file,
            file_content,
            session_id,
            file.filename,
        )
        if not blob_info:
            raise HTTPException(
                status_code=500,
                detail="Failed to store uploaded file"
            )

        # Add to session documents
        current_docs.append({
            "filename": file.filename,
            "content": extraction_result['text'],
            "page_texts": extraction_result.get('page_texts', []),
            "page_count": extraction_result['page_count']
        })

        # Store in Redis with TTL
        await redis_client.setex(
            session_key,
            config.SESSION_TTL_SECONDS,
            json.dumps(current_docs)
        )

        upload_id = await persistence_service.save_upload(
            session_id=session_id,
            filename=file.filename,
            content_type=file.content_type,
            extraction_result=extraction_result,
            blob_info=blob_info,
        )

        logger.info(f"Stored in Redis session: {session_id}")
        logger.info(f"Session now has {len(current_docs)}/{config.MAX_UPLOADS_PER_SESSION} documents")

        return {
            "message": "File uploaded and ready for queries!",
            "filename": file.filename,
            "session_id": session_id,
            "upload_id": upload_id,
            "pages_extracted": extraction_result['page_count'],
            "text_length": len(extraction_result['text']),
            "immediate_access": True,
            "uploads_remaining": config.MAX_UPLOADS_PER_SESSION - len(current_docs)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in upload_document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while uploading document")


@app.post("/api/cleanup-session")
async def cleanup_session(
    request_body: CleanupRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Delete all uploaded documents for a session from Redis.

    Args:
        request_body: Cleanup request with `session_id`.
        authenticated: Authentication dependency guard.
    """
    try:
        session_id = request_body.session_id
        logger.info(f"Cleanup request - Session ID: {session_id}")

        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        redis_client = await get_redis_client()
        session_key = f"session:{session_id}"
        conversation_key = f"conv:{session_id}"
        session_data = await redis_client.get(session_key)

        if session_data:
            session_docs = json.loads(session_data)
            files_count = len(session_docs)
            await redis_client.delete(session_key)
            await redis_client.delete(conversation_key)

            try:
                await persistence_service.delete_session(session_id)
            except Exception as persistence_error:
                logger.warning("Failed to delete persisted session data: %s", persistence_error)

            logger.info(f"Deleted {files_count} documents from Redis session")
            return {
                "message": "Session cleaned up successfully",
                "session_id": session_id,
                "files_deleted": files_count
            }

        logger.warning("Session not found")

        await redis_client.delete(conversation_key)

        try:
            await persistence_service.delete_session(session_id)
        except Exception as persistence_error:
            logger.warning("Failed to delete persisted session data: %s", persistence_error)

        return {
            "message": "No session found",
            "session_id": session_id,
            "files_deleted": 0
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in cleanup_session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while cleaning up session")


@app.get("/api/indexer/status")
async def get_indexer_status(authenticated: bool = Depends(verify_api_key)):
    """Return current Azure Search indexer status and latest execution metadata."""
    try:
        status = await search_service.get_indexer_status()
        return status
    except Exception as e:
        logger.exception(f"Error getting indexer status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching indexer status")


@app.post("/api/indexer/run")
async def run_indexer(authenticated: bool = Depends(verify_api_key)):
    """Manually trigger Azure Search indexer to process newly available documents."""
    try:
        success = await search_service.run_indexer()
        if success:
            return {"message": "Indexer triggered successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to trigger indexer")
    except Exception as e:
        logger.exception(f"Error triggering indexer: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while triggering indexer")


@app.get("/api/health")
async def health_check():
    """
    Basic health check endpoint.

    Verifies Redis connectivity and reports degraded status if Redis is
    unreachable while keeping endpoint responsive.
    """
    health = {"status": "healthy", "redis": "healthy"}

    try:
        redis_client = await get_redis_client()
        await redis_client.ping()
    except Exception as e:
        logger.warning(f"Health check degraded due to Redis issue: {e}")
        health["status"] = "degraded"
        health["redis"] = "unhealthy"

    return health


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)