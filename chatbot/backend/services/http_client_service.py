"""
Shared HTTP client service for Azure SDK/OpenAI calls.

Provides a singleton `httpx.Client` with connection pooling to reduce socket
churn and improve throughput under concurrent request load.
"""

import httpx
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Global shared client
_shared_client: Optional[httpx.Client] = None


def get_shared_http_client() -> httpx.Client:
    """
    Get or create the singleton shared HTTP client.

    Returns:
        httpx.Client: Reusable HTTP client configured with connection pooling,
        HTTP/2 support, and conservative timeout defaults for Azure/OpenAI calls.

    This client is reused across all service instances to reduce socket churn
    and prevent connection exhaustion during concurrent workloads.
    """
    global _shared_client
    
    if _shared_client is None:
        _shared_client = httpx.Client(
            timeout=httpx.Timeout(120.0, connect=10.0),  # 120s total, 10s connect
            limits=httpx.Limits(
                max_connections=500,      # Total connections
                max_keepalive_connections=200,  # Persistent connections
                keepalive_expiry=30.0     # Keep connections alive for 30s
            ),
            http2=True  # Enable HTTP/2 for better performance
        )
        logger.info("Shared HTTP client created with connection pool (max_connections=500)")
    
    return _shared_client


def close_shared_http_client():
    """Close and reset the shared HTTP client during application shutdown."""
    global _shared_client
    if _shared_client:
        _shared_client.close()
        _shared_client = None
        logger.info("Shared HTTP client closed")