from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.credentials import AzureKeyCredential
except Exception:  # Optional dependency until requirements are installed.
    DocumentIntelligenceClient = None
    AnalyzeDocumentRequest = None
    AzureKeyCredential = None

from money_order_validator.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class OcrWord:
    content: str
    confidence: float = 0.0
    polygon: Tuple[float, ...] = ()


@dataclass
class OcrPage:
    page_number: int
    text: str
    angle: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    words: List[OcrWord] = field(default_factory=list)


class AzureDocumentIntelligenceReader:
    def __init__(self) -> None:
        self.endpoint = settings.azure_document_intelligence_endpoint
        self.key = settings.azure_document_intelligence_key

    @property
    def available(self) -> bool:
        return bool(self.endpoint and self.key and DocumentIntelligenceClient and AnalyzeDocumentRequest and AzureKeyCredential)

    async def analyze_pdf(self, content: bytes) -> List[OcrPage]:
        if not self.available:
            return []
        try:
            async with DocumentIntelligenceClient(
                endpoint=self.endpoint.rstrip("/"),
                credential=AzureKeyCredential(self.key),
            ) as client:
                poller = await client.begin_analyze_document(
                    "prebuilt-read",
                    AnalyzeDocumentRequest(bytes_source=content),
                )
                result = await poller.result()
        except Exception as exc:
            logger.warning("Azure Document Intelligence failed; continuing without OCR: %s", exc)
            return []

        pages: List[OcrPage] = []
        for page in sorted(result.pages or [], key=lambda p: p.page_number):
            lines = page.lines or []
            text = "\n".join(line.content for line in lines if getattr(line, "content", None))
            width = getattr(page, "width", None)
            height = getattr(page, "height", None)
            words: List[OcrWord] = []
            for word in page.words or []:
                polygon = getattr(word, "polygon", None) or []
                points: List[float] = []
                for point in polygon:
                    x = getattr(point, "x", None)
                    y = getattr(point, "y", None)
                    if x is None or y is None:
                        continue
                    points.extend(
                        [
                            float(x) / float(width) if width else float(x),
                            float(y) / float(height) if height else float(y),
                        ]
                    )
                words.append(
                    OcrWord(
                        content=str(getattr(word, "content", "") or ""),
                        confidence=float(getattr(word, "confidence", 0.0) or 0.0),
                        polygon=tuple(points),
                    )
                )
            pages.append(
                OcrPage(
                    page_number=page.page_number,
                    text=text,
                    angle=getattr(page, "angle", None),
                    width=width,
                    height=height,
                    words=words,
                )
            )
        logger.info("Azure DI extracted OCR for %d page(s)", len(pages))
        return pages


adi_reader = AzureDocumentIntelligenceReader()


import io
from typing import List

import aiohttp
from PIL import Image

from money_order_validator.evidence import RegionEvidence

logger = logging.getLogger(__name__)


class AzureCustomVisionDetector:
    """Optional trained object detector for complete payment-document regions."""

    @property
    def available(self) -> bool:
        return bool(
            settings.azure_custom_vision_endpoint
            and settings.azure_custom_vision_prediction_key
            and settings.azure_custom_vision_project_id
            and settings.azure_custom_vision_published_name
        )

    async def detect(self, image: Image.Image, page_number: int) -> List[RegionEvidence]:
        if not self.available:
            return []
        endpoint = settings.azure_custom_vision_endpoint.rstrip("/")
        url = (
            f"{endpoint}/customvision/v3.0/Prediction/"
            f"{settings.azure_custom_vision_project_id}/detect/iterations/"
            f"{settings.azure_custom_vision_published_name}/image"
        )
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=settings.custom_vision_timeout_seconds)
            ) as session:
                async with session.post(
                    url,
                    headers={
                        "Prediction-Key": settings.azure_custom_vision_prediction_key,
                        "Content-Type": "application/octet-stream",
                    },
                    data=buffer.getvalue(),
                ) as response:
                    response.raise_for_status()
                    body = await response.json()
        except Exception as exc:
            logger.warning("Azure Custom Vision failed; using heuristic detector: %s", exc)
            return []

        regions: List[RegionEvidence] = []
        for prediction in body.get("predictions") or []:
            confidence = float(prediction.get("probability") or 0.0)
            tag = str(prediction.get("tagName") or "").strip().lower()
            if confidence < settings.custom_vision_min_confidence:
                continue
            box = prediction.get("boundingBox") or {}
            left, top = float(box.get("left") or 0.0), float(box.get("top") or 0.0)
            right = min(1.0, left + float(box.get("width") or 0.0))
            bottom = min(1.0, top + float(box.get("height") or 0.0))
            if right <= left or bottom <= top:
                continue
            crop = image.crop(
                (int(left * image.width), int(top * image.height), int(right * image.width), int(bottom * image.height))
            )
            regions.append(
                RegionEvidence.create(
                    page_number=page_number,
                    image=crop,
                    bbox=(left, top, right, bottom),
                    source=f"custom_vision:{tag}",
                    ocr_text=tag,
                    orientation=90 if crop.height > crop.width else 0,
                    confidence=confidence,
                )
            )
        return regions


custom_vision_detector = AzureCustomVisionDetector()


import asyncio
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from openai import AsyncAzureOpenAI, AsyncOpenAI
except Exception:  # Optional dependency until requirements are installed.
    AsyncAzureOpenAI = None
    AsyncOpenAI = None

from money_order_validator.schemas import TokenUsage

logger = logging.getLogger(__name__)


def is_content_filter_error(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = str(error.get("code") or "").lower() if isinstance(error, dict) else ""
    inner = error.get("innererror") if isinstance(error, dict) else {}
    inner_code = str(inner.get("code") or "").lower() if isinstance(inner, dict) else ""
    message = str(exc).lower()
    return (
        code == "content_filter"
        or inner_code == "responsibleaipolicyviolation"
        or "'code': 'content_filter'" in message
        or "responsibleaipolicyviolation" in message
    )


def _resize_image(image: Image.Image, max_width: int) -> Image.Image:
    img = image.convert("RGB")
    if max_width and img.width > max_width:
        ratio = max_width / float(img.width)
        img = img.resize((max_width, int(img.height * ratio)), Image.Resampling.LANCZOS)
    return img


def image_to_data_url(image: Image.Image, max_width: int, quality: int = 88) -> str:
    img = _resize_image(image, max_width=max_width)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


class LLMClient:
    def __init__(self) -> None:
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_loop: Optional[asyncio.AbstractEventLoop] = None
        self.mode = "none"
        self.model = ""
        self.client: Optional[Any] = None
        self.last_request_metadata: Dict[str, Any] = {}

        if settings.azure_openai_ready and AsyncAzureOpenAI is not None:
            self.client = AsyncAzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint.rstrip("/"),
                api_version=settings.azure_openai_api_version,
                timeout=settings.openai_timeout_seconds,
                max_retries=2,
            )
            self.model = settings.azure_openai_deployment_name
            self.mode = "azure"
        elif settings.openai_api_key and AsyncOpenAI is not None:
            self.client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
                max_retries=2,
            )
            self.model = settings.openai_model_name
            self.mode = "openai"

    @property
    def available(self) -> bool:
        return self.client is not None and bool(self.model)

    def _current_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(max(1, settings.openai_concurrency))
            self._semaphore_loop = loop
        return self._semaphore

    @staticmethod
    def _cache_path(kind: str, payload: Dict[str, Any]) -> Path:
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return Path(settings.evidence_cache_dir) / kind / f"{digest}.json"

    @staticmethod
    def _read_cache(path: Path) -> Optional[Tuple[Dict[str, Any], TokenUsage]]:
        if str(settings.evidence_cache_mode).lower() not in {"read", "readwrite"} or not path.exists():
            return None
        body = json.loads(path.read_text())
        return dict(body.get("response") or {}), TokenUsage(**dict(body.get("usage") or {}))

    @staticmethod
    def _write_cache(path: Path, response: Dict[str, Any], usage: TokenUsage) -> None:
        if str(settings.evidence_cache_mode).lower() not in {"write", "readwrite"}:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"response": response, "usage": usage.model_dump(mode="json")},
                sort_keys=True,
            )
        )

    async def json_vision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image: Image.Image,
        max_width: int,
        detail: str = "high",
        max_completion_tokens: int = 2500,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], TokenUsage]:
        if not self.available:
            return {}, TokenUsage()

        data_url = image_to_data_url(image, max_width=max_width)
        cache_path = self._cache_path(
            "vision",
            {
                "model": self.model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "image_sha256": hashlib.sha256(data_url.encode("ascii")).hexdigest(),
                "detail": detail,
                "max_width": max_width,
                "schema": schema,
            },
        )
        cached = self._read_cache(cache_path)
        if cached:
            return cached
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": detail}},
                ],
            },
        ]
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": (
                {"type": "json_schema", "json_schema": schema}
                if schema
                else {"type": "json_object"}
            ),
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0,
            "seed": settings.openai_seed,
        }
        async with self._current_semaphore():
            try:
                response = await self._create_with_fallback(kwargs)
            except Exception as exc:
                if not is_content_filter_error(exc):
                    raise
                logger.warning("Azure content filter blocked one vision request; continuing with OCR fallback.")
                return {}, TokenUsage()
        content = response.choices[0].message.content or "{}"
        self.last_request_metadata = {
            **self.last_request_metadata,
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "model": getattr(response, "model", None) or self.model,
        }
        parsed = parse_json_object(content)
        usage = TokenUsage.from_openai_usage(getattr(response, "usage", None))
        self._write_cache(cache_path, parsed, usage)
        return parsed, usage

    async def json_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int = 1000,
    ) -> Tuple[Dict[str, Any], TokenUsage]:
        if not self.available:
            return {}, TokenUsage()
        cache_path = self._cache_path(
            "text",
            {"model": self.model, "system_prompt": system_prompt, "user_prompt": user_prompt},
        )
        cached = self._read_cache(cache_path)
        if cached:
            return cached
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0,
            "seed": settings.openai_seed,
        }
        async with self._current_semaphore():
            try:
                response = await self._create_with_fallback(kwargs)
            except Exception as exc:
                if not is_content_filter_error(exc):
                    raise
                logger.warning("Azure content filter blocked one text request; continuing without LLM output.")
                return {}, TokenUsage()
        content = response.choices[0].message.content or "{}"
        self.last_request_metadata = {
            **self.last_request_metadata,
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "model": getattr(response, "model", None) or self.model,
        }
        parsed = parse_json_object(content)
        usage = TokenUsage.from_openai_usage(getattr(response, "usage", None))
        self._write_cache(cache_path, parsed, usage)
        return parsed, usage

    async def _create_with_fallback(self, kwargs: Dict[str, Any]) -> Any:
        assert self.client is not None
        attempts = []

        async def _call(k: Dict[str, Any]) -> Any:
            return await self.client.chat.completions.create(**k)

        variants = [dict(kwargs)]

        no_temp = dict(kwargs)
        no_temp.pop("temperature", None)
        variants.append(no_temp)

        no_seed = dict(no_temp)
        no_seed.pop("seed", None)
        variants.append(no_seed)

        no_json = dict(no_seed)
        no_json.pop("response_format", None)
        variants.append(no_json)

        max_tokens_variant = dict(no_json)
        if "max_completion_tokens" in max_tokens_variant:
            max_tokens_variant["max_tokens"] = max_tokens_variant.pop("max_completion_tokens")
        variants.append(max_tokens_variant)

        last_exc: Optional[Exception] = None
        for index, variant in enumerate(variants):
            signature = tuple(sorted(variant.keys()))
            if signature in attempts:
                continue
            attempts.append(signature)
            try:
                response = await _call(variant)
                self.last_request_metadata = {
                    "fallback_variant": index,
                    "request_parameters": list(signature),
                    "seed_used": "seed" in variant,
                    "temperature_used": "temperature" in variant,
                    "structured_output_used": "response_format" in variant,
                }
                if index:
                    logger.warning(
                        "OpenAI request succeeded with fallback variant %d; seed=%s temperature=%s structured_output=%s",
                        index,
                        "seed" in variant,
                        "temperature" in variant,
                        "response_format" in variant,
                    )
                return response
            except Exception as exc:  # Azure deployments vary on supported params.
                last_exc = exc
                if is_content_filter_error(exc):
                    raise
                msg = str(exc).lower()
                if not any(s in msg for s in ("temperature", "seed", "response_format", "max_completion_tokens", "max_tokens", "unsupported", "unknown parameter")):
                    logger.warning("OpenAI call failed: %s", exc)
                    raise
                logger.info("Retrying OpenAI call with reduced parameters after: %s", exc)
        assert last_exc is not None
        raise last_exc


llm_client = LLMClient()
