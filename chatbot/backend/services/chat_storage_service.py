"""
Durable persistence service for chat history and user uploads.

Uses PostgreSQL when DATABASE_URL is set (production), with optional SQLite
fallback for local development.
"""

import asyncio
import os
import sqlite3
import uuid
import logging
from typing import Dict, List, Optional

import config

try:
    import psycopg
except Exception:  # pragma: no cover - optional in local SQLite mode
    psycopg = None


class PersistenceService:
    """Persist chat and upload records in PostgreSQL or SQLite."""

    def __init__(self):
        self.database_url = config.DATABASE_URL.strip()
        self.use_postgres = bool(self.database_url)
        self.db_path = config.PERSISTENCE_DB_PATH
        self.logger = logging.getLogger(__name__)
        if not self.use_postgres:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.logger.info("Persistence backend: SQLite (%s)", self.db_path)
        else:
            self.logger.info("Persistence backend: PostgreSQL")

        if self.use_postgres and psycopg is None:
            raise RuntimeError("DATABASE_URL is set but psycopg is not installed")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _connect_postgres(self):
        return psycopg.connect(self.database_url)

    def _initialize_sync(self):
        if self.use_postgres:
            return self._initialize_postgres_sync()
        return self._initialize_sqlite_sync()

    def _initialize_sqlite_sync(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS uploads (
                    upload_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT,
                    blob_container TEXT,
                    blob_name TEXT,
                    blob_url TEXT,
                    page_count INTEGER DEFAULT 0,
                    text_length INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS upload_pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    upload_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    page_text TEXT,
                    FOREIGN KEY (upload_id) REFERENCES uploads(upload_id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL,
                    source_filename TEXT,
                    source_type TEXT,
                    citation_number INTEGER,
                    download_url TEXT,
                    FOREIGN KEY (message_id) REFERENCES chat_messages(message_id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_uploads_session_id
                ON uploads(session_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_upload_pages_upload_id
                ON upload_pages(upload_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
                ON chat_messages(session_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at
                ON chat_messages(created_at)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sources_message_id
                ON chat_sources(message_id)
                """
            )
            conn.commit()

    def _initialize_postgres_sync(self):
        with self._connect_postgres() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS uploads (
                        upload_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        content_type TEXT,
                        blob_container TEXT,
                        blob_name TEXT,
                        blob_url TEXT,
                        page_count INTEGER DEFAULT 0,
                        text_length INTEGER DEFAULT 0,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS upload_pages (
                        id BIGSERIAL PRIMARY KEY,
                        upload_id TEXT NOT NULL,
                        page_number INTEGER NOT NULL,
                        page_text TEXT,
                        FOREIGN KEY (upload_id) REFERENCES uploads(upload_id) ON DELETE CASCADE
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        message_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sources (
                        id BIGSERIAL PRIMARY KEY,
                        message_id TEXT NOT NULL,
                        source_filename TEXT,
                        source_type TEXT,
                        citation_number INTEGER,
                        download_url TEXT,
                        FOREIGN KEY (message_id) REFERENCES chat_messages(message_id) ON DELETE CASCADE
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_uploads_session_id
                    ON uploads(session_id)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_upload_pages_upload_id
                    ON upload_pages(upload_id)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
                    ON chat_messages(session_id)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at
                    ON chat_messages(created_at)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_sources_message_id
                    ON chat_sources(message_id)
                    """
                )
            conn.commit()

    async def initialize(self):
        """Initialize persistence tables for configured backend."""
        await asyncio.to_thread(self._initialize_sync)
        if self.use_postgres:
            self.logger.info("Persistence initialized in PostgreSQL")
        else:
            self.logger.info("Persistence initialized at %s", self.db_path)

    def _upsert_session_sync(self, session_id: str):
        if self.use_postgres:
            return self._upsert_session_postgres_sync(session_id)
        return self._upsert_session_sqlite_sync(session_id)

    def _upsert_session_sqlite_sync(self, session_id: str):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id)
                VALUES (?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                """,
                (session_id,),
            )
            conn.commit()

    def _upsert_session_postgres_sync(self, session_id: str):
        with self._connect_postgres() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sessions(session_id)
                    VALUES (%s)
                    ON CONFLICT(session_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                    """,
                    (session_id,),
                )
            conn.commit()

    async def ensure_session(self, session_id: str):
        """Create or touch session row."""
        await asyncio.to_thread(self._upsert_session_sync, session_id)

    def _save_upload_sync(
        self,
        upload_id: str,
        session_id: str,
        filename: str,
        content_type: str,
        blob_container: Optional[str],
        blob_name: Optional[str],
        blob_url: Optional[str],
        extracted_text: str,
        page_texts: List[Dict],
        page_count: int,
    ):
        if self.use_postgres:
            return self._save_upload_postgres_sync(
                upload_id,
                session_id,
                filename,
                content_type,
                blob_container,
                blob_name,
                blob_url,
                extracted_text,
                page_texts,
                page_count,
            )
        return self._save_upload_sqlite_sync(
            upload_id,
            session_id,
            filename,
            content_type,
            blob_container,
            blob_name,
            blob_url,
            extracted_text,
            page_texts,
            page_count,
        )

    def _save_upload_sqlite_sync(
        self,
        upload_id: str,
        session_id: str,
        filename: str,
        content_type: str,
        blob_container: Optional[str],
        blob_name: Optional[str],
        blob_url: Optional[str],
        extracted_text: str,
        page_texts: List[Dict],
        page_count: int,
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO uploads(
                    upload_id, session_id, filename, content_type,
                    blob_container, blob_name, blob_url, page_count, text_length
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    session_id,
                    filename,
                    content_type,
                    blob_container,
                    blob_name,
                    blob_url,
                    page_count,
                    len(extracted_text or ""),
                ),
            )

            conn.executemany(
                """
                INSERT INTO upload_pages(upload_id, page_number, page_text)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        upload_id,
                        page_info.get("page_number", idx + 1),
                        page_info.get("text", ""),
                    )
                    for idx, page_info in enumerate(page_texts or [])
                ],
            )
            conn.commit()

    def _save_upload_postgres_sync(
        self,
        upload_id: str,
        session_id: str,
        filename: str,
        content_type: str,
        blob_container: Optional[str],
        blob_name: Optional[str],
        blob_url: Optional[str],
        extracted_text: str,
        page_texts: List[Dict],
        page_count: int,
    ):
        with self._connect_postgres() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO uploads(
                        upload_id, session_id, filename, content_type,
                        blob_container, blob_name, blob_url, page_count, text_length
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        upload_id,
                        session_id,
                        filename,
                        content_type,
                        blob_container,
                        blob_name,
                        blob_url,
                        page_count,
                        len(extracted_text or ""),
                    ),
                )
                if page_texts:
                    cursor.executemany(
                        """
                        INSERT INTO upload_pages(upload_id, page_number, page_text)
                        VALUES (%s, %s, %s)
                        """,
                        [
                            (
                                upload_id,
                                page_info.get("page_number", idx + 1),
                                page_info.get("text", ""),
                            )
                            for idx, page_info in enumerate(page_texts)
                        ],
                    )
            conn.commit()

    async def save_upload(
        self,
        session_id: str,
        filename: str,
        content_type: str,
        extraction_result: Dict,
        blob_info: Optional[Dict] = None,
    ) -> str:
        """Persist upload metadata and extracted page text."""
        upload_id = str(uuid.uuid4())
        await self.ensure_session(session_id)

        await asyncio.to_thread(
            self._save_upload_sync,
            upload_id,
            session_id,
            filename,
            content_type,
            (blob_info or {}).get("container"),
            (blob_info or {}).get("blob_name"),
            (blob_info or {}).get("blob_url"),
            extraction_result.get("text", ""),
            extraction_result.get("page_texts", []),
            extraction_result.get("page_count", 0),
        )
        return upload_id

    def _save_chat_exchange_sync(
        self,
        session_id: str,
        user_message_id: str,
        assistant_message_id: str,
        query: str,
        answer: str,
        sources: List[Dict],
    ):
        if self.use_postgres:
            return self._save_chat_exchange_postgres_sync(
                session_id,
                user_message_id,
                assistant_message_id,
                query,
                answer,
                sources,
            )
        return self._save_chat_exchange_sqlite_sync(
            session_id,
            user_message_id,
            assistant_message_id,
            query,
            answer,
            sources,
        )

    def _save_chat_exchange_sqlite_sync(
        self,
        session_id: str,
        user_message_id: str,
        assistant_message_id: str,
        query: str,
        answer: str,
        sources: List[Dict],
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages(message_id, session_id, role, content)
                VALUES (?, ?, 'user', ?)
                """,
                (user_message_id, session_id, query),
            )
            conn.execute(
                """
                INSERT INTO chat_messages(message_id, session_id, role, content)
                VALUES (?, ?, 'assistant', ?)
                """,
                (assistant_message_id, session_id, answer),
            )
            conn.executemany(
                """
                INSERT INTO chat_sources(
                    message_id, source_filename, source_type, citation_number, download_url
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        assistant_message_id,
                        source.get("filename"),
                        source.get("type"),
                        source.get("citation_number"),
                        source.get("download_url"),
                    )
                    for source in (sources or [])
                ],
            )
            conn.commit()

    def _save_chat_exchange_postgres_sync(
        self,
        session_id: str,
        user_message_id: str,
        assistant_message_id: str,
        query: str,
        answer: str,
        sources: List[Dict],
    ):
        with self._connect_postgres() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chat_messages(message_id, session_id, role, content)
                    VALUES (%s, %s, 'user', %s)
                    """,
                    (user_message_id, session_id, query),
                )
                cursor.execute(
                    """
                    INSERT INTO chat_messages(message_id, session_id, role, content)
                    VALUES (%s, %s, 'assistant', %s)
                    """,
                    (assistant_message_id, session_id, answer),
                )
                if sources:
                    cursor.executemany(
                        """
                        INSERT INTO chat_sources(
                            message_id, source_filename, source_type, citation_number, download_url
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                assistant_message_id,
                                source.get("filename"),
                                source.get("type"),
                                source.get("citation_number"),
                                source.get("download_url"),
                            )
                            for source in sources
                        ],
                    )
            conn.commit()

    async def save_chat_exchange(
        self,
        session_id: str,
        query: str,
        answer: str,
        sources: List[Dict],
    ):
        """Persist user+assistant messages and assistant source references."""
        await self.ensure_session(session_id)
        await asyncio.to_thread(
            self._save_chat_exchange_sync,
            session_id,
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            query,
            answer,
            sources,
        )

    def _delete_session_sync(self, session_id: str):
        if self.use_postgres:
            return self._delete_session_postgres_sync(session_id)
        return self._delete_session_sqlite_sync(session_id)

    def _delete_session_sqlite_sync(self, session_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    def _delete_session_postgres_sync(self, session_id: str):
        with self._connect_postgres() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
            conn.commit()

    async def delete_session(self, session_id: str):
        """Delete persisted records for a session."""
        await asyncio.to_thread(self._delete_session_sync, session_id)
