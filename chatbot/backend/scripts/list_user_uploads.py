"""
List user-uploaded files from Azure Blob Storage uploads container.

Usage:
    python scripts/list_user_uploads.py
    python scripts/list_user_uploads.py --prefix session-123
"""

import argparse
import os
import sys
from azure.storage.blob import BlobServiceClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _format_size(num_bytes: int) -> str:
    """Return human-readable size string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def list_user_uploads(prefix: str = "") -> None:
    """Print all user-upload blobs and summary counts."""
    if not config.AZURE_STORAGE_CONNECTION_STRING:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not configured")

    blob_service = BlobServiceClient.from_connection_string(
        config.AZURE_STORAGE_CONNECTION_STRING
    )
    container_client = blob_service.get_container_client(
        config.AZURE_UPLOADS_CONTAINER_NAME
    )

    blobs = list(container_client.list_blobs(name_starts_with=prefix or None))

    print(f"Container: {config.AZURE_UPLOADS_CONTAINER_NAME}")
    if prefix:
        print(f"Prefix: {prefix}")
    print("-" * 80)

    total_size = 0
    file_count = 0

    for index, blob in enumerate(blobs, start=1):
        if blob.name.endswith("/"):
            continue
        file_count += 1
        size = int(getattr(blob, "size", 0) or 0)
        total_size += size
        print(f"{index:4d}. {blob.name} | {_format_size(size)}")

    print("-" * 80)
    print(f"Total files: {file_count}")
    print(f"Total size : {_format_size(total_size)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List user-uploaded files from Azure Blob Storage")
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional blob name prefix filter (for example a session id)",
    )
    args = parser.parse_args()

    list_user_uploads(prefix=args.prefix)
