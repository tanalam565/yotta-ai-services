"""
LLM response generation service.

Builds prompts from indexed/uploaded context, manages Redis-backed conversation
history, and returns citation-aware answers from Azure OpenAI.

Key responsibilities:
- Construct system and user prompts with document context
- Manage per-session conversation history via Redis
- Call Azure OpenAI (GPT-5.4) with retry logic
- Parse, renumber, and deduplicate inline [N → Page X] citations
- Return structured response payload with answer, sources, and session_id
"""

from typing import List, Dict, Optional
from openai import AzureOpenAI, RateLimitError, APIConnectionError
import uuid
import re
import json
import asyncio
import logging
import chatbot_config as config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from chatbot_services.redis_service import get_redis_client
from chatbot_services.http_client_service import get_shared_http_client


class LLMService:
    """Build prompts, call Azure OpenAI, and return citation-aware responses."""

    def __init__(self):
        """
        Initialize Azure OpenAI chat client with shared HTTP connection pooling.

        Uses the shared httpx client singleton to avoid socket churn under
        concurrent load. Model and endpoint are pulled from config.
        """
        self.client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            http_client=get_shared_http_client()
        )
        self.model = config.AZURE_OPENAI_DEPLOYMENT_NAME
        self.logger = logging.getLogger(__name__)

    # ── Redis history helpers ─────────────────────────────────────────────────────

    async def _load_history(self, session_id: str) -> list:
        """
        Load prior conversation turns for a session from Redis.

        Returns an empty list if no history exists or Redis is unavailable.
        Citation markers are stripped from stored responses before returning
        to prevent stale citation numbers leaking into new prompts.

        Args:
            session_id: Unique session identifier.

        Returns:
            list: Sanitized list of {"query": str, "response": str} dicts.
        """
        try:
            redis_client = await get_redis_client()
            data = await redis_client.get(f"conv:{session_id}")
            history = json.loads(data) if data else []
            return self._sanitize_history_for_prompt(history)
        except Exception as e:
            self.logger.warning("Redis history load error: %s", e)
            return []

    async def _save_history(self, session_id: str, history: list):
        """
        Save bounded conversation history for a session with configured TTL.

        Trims history to MAX_CONVERSATION_TURNS before saving to prevent
        unbounded Redis growth. Citation markers are stripped prior to storage.

        Args:
            session_id: Unique session identifier.
            history: Full conversation history list including the latest turn.
        """
        try:
            history = self._sanitize_history_for_prompt(history)
            if len(history) > config.MAX_CONVERSATION_TURNS:
                history = history[-config.MAX_CONVERSATION_TURNS:]
            redis_client = await get_redis_client()
            await redis_client.setex(
                f"conv:{session_id}",
                config.SESSION_TTL_SECONDS,
                json.dumps(history)
            )
        except Exception as e:
            self.logger.warning("Redis history save error: %s", e)

    def _sanitize_history_for_prompt(self, history: list) -> list:
        """
        Remove inline citation markers from stored history entries.

        Prevents stale [N → Page X] numbers from prior turns being
        reused or misinterpreted in subsequent prompts after document
        context changes between turns.

        Args:
            history: Raw history list from Redis, may contain citation markers.

        Returns:
            list: Cleaned history with citations removed and whitespace normalized.
        """
        citation_pattern = r'\[(\d+)(?:\s*→\s*Page\s*\d+)?\]'
        sanitized = []

        for entry in history:
            if not isinstance(entry, dict):
                continue

            query = entry.get("query", "")
            response = entry.get("response", "")
            clean_response = re.sub(citation_pattern, '', response)
            clean_response = re.sub(r'\s{2,}', ' ', clean_response).strip()

            sanitized.append({
                "query": query,
                "response": clean_response
            })

        return sanitized

    # ── Prompt builders ───────────────────────────────────────────────────────────

    def _build_system_prompt(self, has_uploads: bool = False) -> str:
        """
        Build the system instruction prompt for the chat completion request.

        Generates a grounding prompt that enforces citation format, bullet-point
        style, and source attribution rules. The prompt differs slightly depending
        on whether user-uploaded documents are present in the context, to guide
        the model toward clearly distinguishing uploaded vs company sources.

        Args:
            has_uploads: True when user-uploaded documents are in the context.

        Returns:
            str: Complete system prompt string.
        """
        base_prompt = """You are an AI assistant for YottaReal property management software, helping leasing agents, property managers, and district managers retrieve information.

Your role:
- Answer questions based ONLY on the provided context from documents
- Be thorough and detailed in your responses
- If information is not in the provided context, clearly state that you don't have that information
- Focus on practical, actionable information

FORMATTING REQUIREMENTS (CRITICAL):
- Do NOT use ** for bold text or any Markdown formatting
- DO use bullet points with this EXACT format:
  
  Main topic:
  - Bullet point 1 with details
  - Bullet point 2 with details
  - Bullet point 3 with details

- Each bullet point should be on its OWN line or paragraph for clarity.
- Use dashes (-) for bullet points

CRITICAL CITATION REQUIREMENT WITH PAGE NUMBERS:
When you reference information from a document, you MUST cite it using this format:
[N → Page X] where N is the document number and X is the actual page number from the PDF

Example: "According to the Move-Out Policy [1 → Page 3], residents must provide 60 days notice."

Guidelines:
- Prioritize accuracy and completeness
- Use bullet points on separate lines for easy reading
- Include relevant policy numbers or section references when available
- Provide detailed explanations with context
- For ambiguous queries, ask clarifying questions
- Always ground your answers in the provided documents
- ALWAYS include [N → Page X] citations when referencing specific information from a document
- NEVER add citations to conversational responses (greetings, farewells, clarifying questions, or any reply where you are not directly quoting or summarizing document content)
- Make responses thorough and informative"""

        if has_uploads:
            base_prompt += """

SOURCE ATTRIBUTION:
- When referencing UPLOADED documents, say "According to your uploaded document [N → Page X]..." or "In [document name] [N → Page X]..."
- When referencing COMPANY or BLOB STORAGE documents (policies, handbooks), say "According to [policy/handbook name] [N → Page X]..." or "Company policy [N → Page X] states..."
- Be clear about which source each piece of information comes from
- If there are multiple uploaded documents and the query is ambiguous, describe ALL of them with their [N → Page X] citations
- Provide comprehensive details from the uploaded documents in bullet format"""
        else:
            base_prompt += """

SOURCE ATTRIBUTION:
- When referencing information, naturally mention the source with [N → Page X] citation (e.g., "According to the Move-Out Policy [1 → Page 3]..." or "As stated in the Team Member Handbook [2 → Page 15]...")
- Provide comprehensive information from the cited documents in bullet format"""

        return base_prompt

    def _build_prompt(self, query: str, context: List[Dict], has_uploads: bool = False) -> tuple:
        """
        Build the user-facing prompt and document mapping for citation renumbering.

        Separates context into uploaded and company document sections, assigns
        sequential document numbers, and tracks page numbers per document for
        later citation renumbering. Company document content is capped at 10,000
        characters to stay within token limits.

        Args:
            query: The user's question text.
            context: List of document chunk dicts with keys: content, filename,
                     source_type, page_number, download_url.
            has_uploads: True when uploaded documents are present (affects section headers).

        Returns:
            tuple[str, dict]: The full prompt string and a doc_mapping dict keyed
                              by document number with filename, type, download_url,
                              and pages metadata.
        """
        uploaded_docs = [doc for doc in context if doc.get("source_type") == "uploaded"]
        company_docs = [doc for doc in context if doc.get("source_type") == "company"]

        context_text = ""
        doc_number = 1
        doc_mapping = {}

        if uploaded_docs:
            context_text += "=== UPLOADED DOCUMENTS (User's Files) ===\n"
            for doc in uploaded_docs:
                page_num = doc.get('page_number', 1)
                context_text += f"\n[Document {doc_number} - Page {page_num}: {doc['filename']}]\n"
                if doc_number not in doc_mapping:
                    doc_mapping[doc_number] = {
                        "filename": doc['filename'],
                        "type": "uploaded",
                        "download_url": doc.get('download_url'),
                        "pages": set()
                    }
                doc_mapping[doc_number]["pages"].add(page_num)
                context_text += f"{doc['content']}\n"
                context_text += f"(End of Document {doc_number} - Page {page_num})\n"
                doc_number += 1

        if company_docs:
            if uploaded_docs:
                context_text += "\n" + "=" * 60 + "\n\n"
            context_text += "=== COMPANY DOCUMENTS (Policies, Handbooks, Procedures) ===\n"
            for doc in company_docs:
                page_num = doc.get('page_number', 1)
                context_text += f"\n[Document {doc_number} - Page {page_num}: {doc['filename']}]\n"
                if doc_number not in doc_mapping:
                    doc_mapping[doc_number] = {
                        "filename": doc['filename'],
                        "type": "company",
                        "download_url": doc.get('download_url'),
                        "pages": set()
                    }
                doc_mapping[doc_number]["pages"].add(page_num)
                content = doc['content'][:10000]
                context_text += f"{content}\n"
                if len(doc['content']) > 10000:
                    context_text += f"... (content truncated, original length: {len(doc['content'])} chars)\n"
                context_text += f"(End of Document {doc_number} - Page {page_num})\n"
                doc_number += 1

        prompt = f"""Context from documents:

{context_text}

User question: {query}

Answer (use bullet points on separate lines with [N → Page X] citations):"""

        return prompt, doc_mapping

    # ── Citation post-processing ──────────────────────────────────────────────────

    def _extract_citations_and_renumber(self, response_text: str, doc_mapping: Dict) -> tuple:
        """
        Normalize citation numbering in the model response and build source metadata.

        After the model generates a response, document numbers may not start from 1
        or may reference the same filename multiple times. This method:
        1. Normalizes any placeholder [N → Page X] template citations
        2. Expands grouped citations like [1; 2; 3] into individual brackets
        3. Deduplicates citations by filename
        4. Renumbers citations sequentially starting from 1
        5. Builds the final sources list with icons and download URLs

        Args:
            response_text: Raw model response text containing citation markers.
            doc_mapping: Document number → metadata mapping from _build_prompt.

        Returns:
            tuple[str, list]: Updated response text with renumbered citations,
                              and a list of unique source dicts for the UI.
        """
        response_text = self._normalize_placeholder_citations(response_text, doc_mapping)
        response_text = self._expand_grouped_citations(response_text)

        citation_pattern = r'\[(\d+)(?:\s*→\s*Page\s*(\d+))?\]'
        matches = re.finditer(citation_pattern, response_text)

        cited_docs = {}
        for match in matches:
            doc_num = int(match.group(1))
            page_num = match.group(2)
            if doc_num not in cited_docs:
                cited_docs[doc_num] = set()
            if page_num:
                cited_docs[doc_num].add(int(page_num))

        unique_sources = {}
        new_num = 1

        for doc_num in sorted(cited_docs.keys()):
            if doc_num in doc_mapping:
                doc_info = doc_mapping[doc_num]
                filename = doc_info['filename']
                if filename not in unique_sources:
                    unique_sources[filename] = {
                        "new_num": new_num,
                        "type": doc_info["type"],
                        "download_url": doc_info.get("download_url"),
                        "old_nums": []
                    }
                    new_num += 1
                unique_sources[filename]["old_nums"].append(doc_num)

        renumber_map = {}
        for filename, info in unique_sources.items():
            for old_num in info["old_nums"]:
                renumber_map[old_num] = info["new_num"]

        def replace_citation(match):
            """Map old citation numbers to deduplicated numbers, preserving page refs."""
            old_num = int(match.group(1))
            page_num = match.group(2)
            if old_num in renumber_map:
                new_num = renumber_map[old_num]
                if page_num:
                    return f"[{new_num} → Page {page_num}]"
                else:
                    return f"[{new_num}]"
            return match.group(0)

        updated_text = re.sub(citation_pattern, replace_citation, response_text)

        sources = []
        for filename, info in sorted(unique_sources.items(), key=lambda x: x[1]["new_num"]):
            icon = "📤" if info["type"] == "uploaded" else "📁"
            sources.append({
                "filename": f"{icon} {filename}",
                "type": info["type"],
                "download_url": info.get("download_url"),
                "citation_number": info["new_num"]
            })

        return updated_text, sources

    def _normalize_placeholder_citations(self, response_text: str, doc_mapping: Dict) -> str:
        """
        Replace literal template citations [N → Page X] with concrete numeric citations.

        The model occasionally echoes the example citation format from the system
        prompt verbatim (using the literal letter N or X). This method replaces
        those placeholders with the actual first uploaded document number and page.

        Args:
            response_text: Model response possibly containing [N → Page X] literals.
            doc_mapping: Document number → metadata mapping.

        Returns:
            str: Response text with placeholder citations replaced by real ones.
        """
        if not doc_mapping:
            return response_text

        uploaded_doc_numbers = [
            doc_num for doc_num, info in doc_mapping.items()
            if info.get("type") == "uploaded"
        ]
        fallback_doc_num = min(uploaded_doc_numbers) if uploaded_doc_numbers else min(doc_mapping.keys())

        fallback_pages = doc_mapping.get(fallback_doc_num, {}).get("pages") or {1}
        fallback_page = min(fallback_pages)

        template_pattern = r'\[(N|n)(?:\s*→\s*Page\s*(X|x|\d+))?\]'

        def replace_template(match):
            raw_page = match.group(2)
            if raw_page and raw_page.isdigit():
                page_number = raw_page
            else:
                page_number = str(fallback_page)
            return f"[{fallback_doc_num} → Page {page_number}]"

        return re.sub(template_pattern, replace_template, response_text)

    def _expand_grouped_citations(self, response_text: str) -> str:
        """
        Expand grouped citation blocks into individual bracketed citations.

        Handles cases where the model groups multiple citations in one block,
        e.g. [1 → Page 51; 2 → Page 1; 3 → Page 4] → [1 → Page 51] [2 → Page 1] [3 → Page 4].

        Args:
            response_text: Model response possibly containing grouped citations.

        Returns:
            str: Response text with all grouped citations expanded.
        """
        grouped_pattern = r'\[((?:\s*\d+\s*(?:→\s*Page\s*\d+)?\s*[;,]\s*)+\s*\d+\s*(?:→\s*Page\s*\d+)?\s*)\]'

        def replace_group(match):
            parts = [part.strip() for part in re.split(r'[;,]', match.group(1)) if part.strip()]
            return " ".join(f"[{part}]" for part in parts)

        return re.sub(grouped_pattern, replace_group, response_text)

    def _clean_response(self, response_text: str) -> str:
        """
        Remove undesired markdown formatting and trim response text.

        Strips any ** bold markers the model may produce despite the system
        prompt instructing otherwise, then trims surrounding whitespace.

        Args:
            response_text: Raw model output string.

        Returns:
            str: Cleaned response text.
        """
        cleaned = re.sub(r'\*\*', '', response_text)
        return cleaned.strip()

    # ── OpenAI call with tenacity retry ──────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3)
    )
    def _call_openai_sync(self, messages: list) -> str:
        """
        Execute a synchronous Azure OpenAI chat completion with retry support.

        Uses max_completion_tokens (required by GPT-5.4) instead of the legacy
        max_tokens parameter. Retries automatically on rate limit or connection
        errors with exponential backoff up to 3 attempts.

        This method is synchronous and must be called via asyncio.to_thread
        to avoid blocking the event loop.

        Args:
            messages: Full message list in OpenAI chat format, including system
                      prompt, conversation history, and the current user message.

        Returns:
            str: The assistant's response content string.

        Raises:
            RateLimitError: If all retry attempts are exhausted due to rate limiting.
            APIConnectionError: If all retry attempts fail due to connectivity issues.
        """
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "timeout": config.REQUEST_TIMEOUT_SECONDS,
            "max_completion_tokens": 800,
        }

        response = self.client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content

    async def _generate_azure_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list
    ) -> str:
        """
        Build the full message sequence and execute the Azure OpenAI chat completion.

        Assembles the messages list from the system prompt, prior conversation
        history turns, and the current user prompt, then dispatches the blocking
        API call to a thread pool to keep the event loop free.

        Args:
            system_prompt: System instruction string from _build_system_prompt.
            user_prompt: User-facing prompt string from _build_prompt.
            history: Sanitized conversation history list of {"query", "response"} dicts.

        Returns:
            str: Raw assistant response content from Azure OpenAI.
        """
        messages = [{"role": "system", "content": system_prompt}]

        # Cap each history response to 600 chars and total history to 3,000 chars
        # to keep prompt size stable as conversations grow.
        MAX_RESPONSE_CHARS = 600
        MAX_HISTORY_CHARS = 3000
        total_history_chars = 0
        trimmed_history = []
        for msg in reversed(history):
            response_text = msg["response"][:MAX_RESPONSE_CHARS]
            entry_chars = len(msg["query"]) + len(response_text)
            if total_history_chars + entry_chars > MAX_HISTORY_CHARS:
                break
            trimmed_history.insert(0, {"query": msg["query"], "response": response_text})
            total_history_chars += entry_chars

        for msg in trimmed_history:
            messages.append({"role": "user", "content": msg["query"]})
            messages.append({"role": "assistant", "content": msg["response"]})

        messages.append({"role": "user", "content": user_prompt})

        self.logger.info("Including %s previous exchanges in context (budget: %s/%s chars)",
                         len(trimmed_history), total_history_chars, MAX_HISTORY_CHARS)

        return await asyncio.to_thread(self._call_openai_sync, messages)

    # ── Main entry point ──────────────────────────────────────────────────────────

    async def generate_response(
        self,
        query: str,
        context: List[Dict],
        session_id: Optional[str] = None,
        has_uploads: bool = False,
        is_comparison: bool = False,
        prefetched_history: Optional[list] = None,
    ) -> Dict:
        """
        Generate a citation-aware answer for a user query.

        Orchestrates the full response pipeline:
        1. Load conversation history from Redis
        2. Build system and user prompts with document context
        3. Log prompt size statistics for monitoring
        4. Call Azure OpenAI and handle errors gracefully
        5. Clean and post-process citations in the response
        6. Persist updated history back to Redis
        7. Return structured response payload

        Args:
            query: The user's question text.
            context: Retrieved document chunks from search and/or session uploads.
            session_id: Optional existing session ID. A new UUID is generated if absent.
            has_uploads: True when user-uploaded documents are present in context.
            is_comparison: Reserved for future comparison-mode prompt branching.

        Returns:
            Dict: Payload with keys:
                - "answer" (str): Citation-annotated response text.
                - "sources" (list): Deduplicated source list with filename, type,
                  download_url, and citation_number.
                - "session_id" (str): The active session identifier.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        # Use caller-supplied history to avoid an extra Redis round trip.
        history = prefetched_history if prefetched_history is not None else await self._load_history(session_id)

        system_prompt = self._build_system_prompt(has_uploads)
        user_prompt, doc_mapping = self._build_prompt(query, context, has_uploads)

        total_chars = len(user_prompt)
        estimated_tokens = total_chars // 4
        uploaded_chars = sum(len(doc['content']) for doc in context if doc.get('source_type') == 'uploaded')
        company_chars = sum(min(len(doc['content']), 10000) for doc in context if doc.get('source_type') == 'company')

        self.logger.info(
            "Prompt stats: total_chars=%s, est_tokens=%s, uploaded_chars=%s, company_chars=%s",
            f"{total_chars:,}",
            f"{estimated_tokens:,}",
            f"{uploaded_chars:,}",
            f"{company_chars:,}",
        )

        try:
            response = await self._generate_azure_openai(system_prompt, user_prompt, history)

            cleaned_response = self._clean_response(response)
            updated_response, sources = self._extract_citations_and_renumber(cleaned_response, doc_mapping)

            self.logger.info(
                "Generated response with inline citations; documents_provided=%s, unique_cited=%s",
                len(doc_mapping),
                len(sources),
            )
            if not sources:
                self.logger.warning("No documents cited in response")

            history.append({"query": query, "response": updated_response})
            await self._save_history(session_id, history)

            return {
                "answer": updated_response,
                "sources": sources,
                "session_id": session_id
            }

        except Exception as e:
            self.logger.exception("LLM generation error: %s", e)
            return {
                "answer": "I apologize, but I encountered an error processing your request.",
                "sources": [],
                "session_id": session_id
            }