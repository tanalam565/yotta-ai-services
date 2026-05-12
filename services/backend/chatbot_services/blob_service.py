"""
Azure Blob Storage helper service.

Generates short-lived SAS download URLs for indexed blob files so the frontend
can provide secure, time-limited document access.
"""

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import ResourceExistsError
from datetime import datetime, timedelta
from typing import Optional
import urllib.parse
import os
import uuid
import logging
import chatbot_config as config

class BlobService:
    """Generate secure, time-limited download URLs for blob documents."""

    def __init__(self):
        """Initialize Azure Blob service client using configured connection details."""
        self.blob_service_client = BlobServiceClient.from_connection_string(
            config.AZURE_STORAGE_CONNECTION_STRING
        )
        self.container_name = config.AZURE_STORAGE_CONTAINER_NAME
        self.uploads_container_name = config.AZURE_UPLOADS_CONTAINER_NAME
        self.logger = logging.getLogger(__name__)

    def upload_user_file(self, file_content: bytes, session_id: str, filename: str) -> Optional[dict]:
        """
        Upload a user-provided file to the dedicated uploads container.

        Returns:
            Optional[dict]: Uploaded blob metadata (container/name/url), or None on error.
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.uploads_container_name)
            try:
                container_client.create_container()
            except ResourceExistsError:
                pass

            safe_filename = os.path.basename(filename or "upload.bin")
            blob_name = f"{session_id}/{uuid.uuid4()}_{safe_filename}"
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(file_content, overwrite=False)

            return {
                "container": self.uploads_container_name,
                "blob_name": blob_name,
                "blob_url": blob_client.url,
            }
        except Exception as e:
            self.logger.error("Error uploading user file %s: %s", filename, e)
            return None
    
    def generate_download_url(self, blob_name: str, expiry_hours: int = 1) -> str:
        """
        Generate a time-limited SAS URL for downloading a blob.

        Args:
            blob_name: Blob filename/path in the configured container.
            expiry_hours: SAS token validity duration in hours.

        Returns:
            Optional[str]: Download URL if generated successfully, otherwise None.
        """
        try:
            # URL decode the blob name first (Azure stores with + as space)
            blob_name = urllib.parse.unquote(blob_name)
            
            # Parse connection string
            conn_parts = dict(
                item.split('=', 1) 
                for item in config.AZURE_STORAGE_CONNECTION_STRING.split(';') 
                if '=' in item
            )
            account_name = conn_parts.get('AccountName')
            account_key = conn_parts.get('AccountKey')
            
            if not account_name or not account_key:
                return None
            
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
                # To force download, uncomment the next line and comment the following line
                content_disposition='attachment'  
                # To view in new tab, use inline
                # content_disposition='inline'
            )
            
            # URL encode the blob name for the URL
            encoded_blob_name = urllib.parse.quote(blob_name, safe='')
            
            # Return full URL with SAS token
            return f"https://{account_name}.blob.core.windows.net/{self.container_name}/{encoded_blob_name}?{sas_token}"
            
        except Exception as e:
            self.logger.error("Error generating download URL for %s: %s", blob_name, e)
            return None