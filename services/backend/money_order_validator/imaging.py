from __future__ import annotations

import re
from typing import Iterable, List

from money_order_validator.settings import settings


IMPORTANT_PATTERNS = [
    r"PAY\s+EXACTLY",
    r"PAY\s+TO\s+THE\s+ORDER",
    r"PAYABLE\s+TO",
    r"PURCHASER|REMITTER|DRAWER|SENDER",
    r"MONEY\s*ORDER|CASHIER|CHECK\s+NO|SERIAL",
    r"WESTERN\s+UNION|MONEYGRAM|INTERMEX|FIDELITY|BARRI|DOLEX|PLS|CHASE|JPMORGAN|WELLS\s+FARGO|PROSPERITY",
    r"\$\s*\d|\*+\s*\$?\d",
    r"\b\d{2}[-\s]?\d{7,12}\b|\b\d{9,14}\b",
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b20\d{2}-\d{2}-\d{2}\b",
    r"APT|APARTMENT|UNIT|SUITE|#\s*\d",
    r"BATCH\s*#|BATCH\s+AMOUNT|ACTUAL\s+ITEMS|TOTALS?\b|DEPOSIT\s+ACCOUNT|TRANSACTION\s+ID",
    r"MICR|ROUTING|ACCOUNT",
]

# Compiled once at import; compact_ocr_context runs per page.
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in IMPORTANT_PATTERNS]


def _clean_line(line: str) -> str:
    line = re.sub(r"[ \t]+", " ", line.strip())
    return line


def dedupe_keep_order(lines: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for line in lines:
        key = re.sub(r"\W+", "", line).upper()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def compact_ocr_context(text: str, max_chars: int | None = None) -> str:
    if not text:
        return ""
    max_chars = max_chars or settings.ocr_context_max_chars
    raw_lines = [_clean_line(x) for x in text.splitlines()]
    raw_lines = [x for x in raw_lines if len(x) >= 2]

    keep: List[str] = []
    for idx, line in enumerate(raw_lines):
        if any(p.search(line) for p in _COMPILED_PATTERNS):
            # Add neighbor lines because OCR splits values across lines/columns.
            if idx > 0:
                keep.append(raw_lines[idx - 1])
            keep.append(line)
            if idx + 1 < len(raw_lines):
                keep.append(raw_lines[idx + 1])

    if not keep:
        keep = raw_lines[:35]

    keep = dedupe_keep_order(keep)
    text_out = "\n".join(keep)
    if len(text_out) <= max_chars:
        return text_out

    # Preserve beginning and last numeric-heavy lines.
    head = []
    total = 0
    for line in keep:
        if total + len(line) + 1 > max_chars:
            break
        head.append(line)
        total += len(line) + 1
    return "\n".join(head)


import asyncio
import io
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import List

import fitz  # PyMuPDF
from PIL import Image



class PdfRenderer:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._region_cache: dict[str, Image.Image] = {}
        self.region_cache_hits = 0
        self.region_cache_misses = 0

    async def render(self, pdf_content: bytes, dpi: int | None = None) -> List[Image.Image]:
        dpi = dpi or settings.pdf_render_dpi
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._render_sync, pdf_content, dpi)

    @staticmethod
    def _render_sync(pdf_content: bytes, dpi: int) -> List[Image.Image]:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        images: List[Image.Image] = []
        try:
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                images.append(img)
        finally:
            doc.close()
        return images

    async def render_page(self, pdf_content: bytes, page_number: int, dpi: int) -> Image.Image:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._render_page_sync, pdf_content, page_number, dpi
        )

    @staticmethod
    def _render_page_sync(pdf_content: bytes, page_number: int, dpi: int) -> Image.Image:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        try:
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
            return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        finally:
            doc.close()

    async def render_region(
        self,
        pdf_content: bytes,
        page_number: int,
        bbox: tuple[float, float, float, float],
        dpi: int,
        orientation: int = 0,
    ) -> Image.Image:
        key = hashlib.sha256(
            pdf_content
            + f":{page_number}:{bbox}:{dpi}:{orientation}".encode("ascii")
        ).hexdigest()
        cached = self._region_cache.get(key)
        if cached is not None:
            self.region_cache_hits += 1
            return cached.copy()
        self.region_cache_misses += 1
        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(
            self._executor,
            self._render_region_sync,
            pdf_content,
            page_number,
            bbox,
            dpi,
            orientation,
        )
        if len(self._region_cache) >= 128:
            self._region_cache.pop(next(iter(self._region_cache)))
        self._region_cache[key] = image.copy()
        return image

    @staticmethod
    def _render_region_sync(
        pdf_content: bytes,
        page_number: int,
        bbox: tuple[float, float, float, float],
        dpi: int,
        orientation: int,
    ) -> Image.Image:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        try:
            page = doc.load_page(page_number - 1)
            x0, y0, x1, y1 = bbox
            rect = page.rect
            clip = fitz.Rect(
                rect.x0 + x0 * rect.width,
                rect.y0 + y0 * rect.height,
                rect.x0 + x1 * rect.width,
                rect.y0 + y1 * rect.height,
            )
            pix = page.get_pixmap(
                matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
                clip=clip,
                alpha=False,
            )
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            return image.rotate(orientation, expand=True) if orientation else image
        finally:
            doc.close()


pdf_renderer = PdfRenderer()


import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps

from money_order_validator.evidence import RegionEvidence


@dataclass(frozen=True)
class SpatialEvidenceNode:
    kind: str
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


def _node_kind(text: str) -> Optional[str]:
    upper = text.upper().strip()
    digits = re.sub(r"\D", "", upper)
    if re.search(r"\bPAY\s+TO\b|\bORDER\s+OF\b", upper):
        return "payee"
    if re.search(r"\bCHECK\b|\bSERIAL\b|\bDOCUMENT\b|\bNO\.?\b", upper) and digits:
        return "document_number"
    if re.search(r"\$\s*[\d,]+(?:\.\d{2})?", upper) or re.fullmatch(r"[\d,]+\.\d{2}", upper):
        return "amount_box"
    if re.search(r"\bDOLLARS?\b|\bCENTS?\b|\bHUNDRED\b|\bTHOUSAND\b", upper):
        return "amount_words"
    if len(digits) >= 4:
        return "micr"
    return None


def _word_xy(word: object):
    """Normalized (xs, ys) of a word polygon, or None if missing/out of range."""
    polygon = tuple(getattr(word, "polygon", ()) or ())
    if len(polygon) < 4:
        return None
    xs, ys = polygon[0::2], polygon[1::2]
    return (xs, ys) if all(0 <= value <= 1 for value in (*xs, *ys)) else None


def spatial_evidence_graph(words: list) -> List[List[SpatialEvidenceNode]]:
    """Build connected instrument-evidence subgraphs from Azure word polygons."""
    nodes: List[SpatialEvidenceNode] = []
    for word in words or []:
        text = str(getattr(word, "content", "") or "")
        kind = _node_kind(text)
        xy = _word_xy(word)
        if not kind or xy is None:
            continue
        xs, ys = xy
        nodes.append(
            SpatialEvidenceNode(
                kind,
                text,
                (min(xs), min(ys), max(xs), max(ys)),
                float(getattr(word, "confidence", 0.0) or 0.0),
            )
        )
    if not nodes:
        return []

    adjacency: List[set[int]] = [set() for _ in nodes]
    for left_index, left in enumerate(nodes):
        lx, ly = left.center
        for right_index in range(left_index + 1, len(nodes)):
            right = nodes[right_index]
            rx, ry = right.center
            horizontal = abs(lx - rx)
            vertical = abs(ly - ry)
            aligned = horizontal < 0.30 and vertical < 0.28
            same_line = vertical < 0.045 and horizontal < 0.28
            if aligned or same_line:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    groups: List[List[SpatialEvidenceNode]] = []
    unseen = set(range(len(nodes)))
    while unseen:
        stack = [unseen.pop()]
        indexes: List[int] = []
        while stack:
            current = stack.pop()
            indexes.append(current)
            neighbors = adjacency[current] & unseen
            unseen -= neighbors
            stack.extend(neighbors)
        group = [nodes[index] for index in indexes]
        kinds = {node.kind for node in group}
        identity = bool(kinds & {"document_number", "micr"})
        amount = bool(kinds & {"amount_box", "amount_words"})
        if identity and amount:
            groups.append(group)
    return groups


def evidence_graph_region_proposals(
    image: Image.Image,
    words: list,
    *,
    page_number: int,
    max_regions: int = 12,
) -> List[RegionEvidence]:
    """Convert graph-local evidence into padded, non-leaking proposals."""
    output: List[RegionEvidence] = []
    for graph in spatial_evidence_graph(words):
        box = (
            max(0.0, min(node.bbox[0] for node in graph) - 0.08),
            max(0.0, min(node.bbox[1] for node in graph) - 0.14),
            min(1.0, max(node.bbox[2] for node in graph) + 0.08),
            min(1.0, max(node.bbox[3] for node in graph) + 0.14),
        )
        text = "\n".join(f"{node.kind}: {node.text}" for node in graph)
        output.append(
            RegionEvidence.create(
                page_number=page_number,
                image=image.crop((int(box[0] * image.width), int(box[1] * image.height), int(box[2] * image.width), int(box[3] * image.height))),
                bbox=box,
                source="evidence_graph",
                ocr_text=text,
                orientation=90 if (box[3] - box[1]) > (box[2] - box[0]) else 0,
                confidence=min(0.99, sum(node.confidence for node in graph) / len(graph)),
            )
        )
    return merge_region_proposals(output, iou_threshold=0.58)[:max_regions]


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (
        max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        + max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def merge_region_proposals(regions: List[RegionEvidence], *, iou_threshold: float = 0.42) -> List[RegionEvidence]:
    """Geometrically merge overlapping proposals before any vision call."""
    output: List[RegionEvidence] = []
    for region in sorted(regions, key=lambda row: row.confidence, reverse=True):
        match = next((kept for kept in output if _bbox_iou(region.bbox, kept.bbox) >= iou_threshold), None)
        if match is None:
            output.append(region)
            continue
        match.bbox = (
            min(match.bbox[0], region.bbox[0]),
            min(match.bbox[1], region.bbox[1]),
            max(match.bbox[2], region.bbox[2]),
            max(match.bbox[3], region.bbox[3]),
        )
        match.confidence = max(match.confidence, region.confidence)
        match.evidence.update(region.evidence)
    return output


def _strong_valley(profile: np.ndarray, *, edge_fraction: float = 0.18) -> Optional[int]:
    """Return a strong internal whitespace valley, avoiding document edges."""
    if profile.size < 20:
        return None
    smooth_width = max(3, int(profile.size * 0.015))
    smooth = np.convolve(profile, np.ones(smooth_width) / smooth_width, mode="same")
    start, end = int(profile.size * edge_fraction), int(profile.size * (1 - edge_fraction))
    if end <= start:
        return None
    internal = smooth[start:end]
    index = int(np.argmin(internal)) + start
    active = smooth[smooth > 0]
    baseline = float(np.percentile(active, 55)) if active.size else 0.0
    return index if baseline > 0 and smooth[index] <= baseline * 0.18 else None


def split_region_proposals(
    image: Image.Image,
    regions: List[RegionEvidence],
    *,
    max_depth: int = 2,
) -> List[RegionEvidence]:
    """Recursively split oversized proposals at strong two-axis whitespace valleys."""
    width, height = image.size

    def split(region: RegionEvidence, depth: int) -> List[RegionEvidence]:
        x0, y0, x1, y1 = region.bbox
        px = (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))
        crop = image.crop(px).convert("L")
        ink = np.asarray(crop) < 225
        if depth >= max_depth or ink.size == 0:
            return [region]
        area = (x1 - x0) * (y1 - y0)
        aspect = max(crop.width, crop.height) / max(1, min(crop.width, crop.height))
        if area < 0.10 and aspect < 4.5:
            return [region]
        vertical = _strong_valley(ink.mean(axis=0))
        horizontal = _strong_valley(ink.mean(axis=1))
        # Prefer the split that creates document-like children.
        axis = "x" if vertical is not None and crop.width >= crop.height else "y"
        point = vertical if axis == "x" else horizontal
        if point is None:
            axis, point = ("y", horizontal) if horizontal is not None else ("x", vertical)
        if point is None:
            return [region]
        fraction = point / (crop.width if axis == "x" else crop.height)
        if axis == "x":
            boxes = [(x0, y0, x0 + (x1 - x0) * fraction, y1), (x0 + (x1 - x0) * fraction, y0, x1, y1)]
        else:
            boxes = [(x0, y0, x1, y0 + (y1 - y0) * fraction), (x0, y0 + (y1 - y0) * fraction, x1, y1)]
        children: List[RegionEvidence] = []
        for box in boxes:
            bx0, by0, bx1, by1 = box
            child = RegionEvidence.create(
                page_number=region.page_number,
                image=image.crop((int(bx0 * width), int(by0 * height), int(bx1 * width), int(by1 * height))),
                bbox=box,
                source=f"{region.source}_valley_split",
                orientation=region.orientation,
                confidence=min(0.99, region.confidence + 0.08),
            )
            children.extend(split(child, depth + 1))
        return children

    return [child for region in regions for child in split(region, 0)]


def document_region_proposals(
    image: Image.Image,
    *,
    page_number: int,
    max_regions: int = 12,
) -> List[RegionEvidence]:
    """Return document proposals in original normalized page coordinates."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    ink = (gray < 225).astype("uint8") * 255
    joined = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(15, int(width * 0.018)), max(15, int(height * 0.055)))
        ),
        iterations=2,
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proposals: List[RegionEvidence] = []
    page_area = float(width * height)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / page_area
        aspect = max(w, h) / max(1, min(w, h))
        if not (0.025 <= area_ratio <= 0.70 and 1.05 <= aspect <= 7.0):
            continue
        mx, my = int(w * 0.10), int(h * 0.18)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(width, x + w + mx), min(height, y + h + my)
        bbox = (x0 / width, y0 / height, x1 / width, y1 / height)
        orientation = 90 if h > w else 0
        confidence = min(0.95, 0.45 + area_ratio + min(aspect, 4.0) * 0.08)
        proposals.append(
            RegionEvidence.create(
                page_number=page_number,
                image=image.crop((x0, y0, x1, y1)),
                bbox=bbox,
                source="component_geometry",
                orientation=orientation,
                confidence=confidence,
            )
        )
    return merge_region_proposals(proposals)[:max_regions]


def ocr_cluster_region_proposals(
    image: Image.Image,
    words: list,
    *,
    page_number: int,
    max_regions: int = 12,
) -> List[RegionEvidence]:
    """Cluster spatial OCR words in two dimensions into document proposals."""
    points: List[Tuple[float, float, float, float, str]] = []
    for word in words or []:
        xy = _word_xy(word)
        if xy is None:
            continue
        xs, ys = xy
        points.append((min(xs), min(ys), max(xs), max(ys), str(getattr(word, "content", "") or "")))
    if not points:
        return []
    groups: List[List[Tuple[float, float, float, float, str]]] = []
    for point in sorted(points, key=lambda row: (row[1], row[0])):
        match = next(
            (
                group
                for group in groups
                if any(
                    abs((point[0] + point[2]) / 2 - (item[0] + item[2]) / 2) < 0.34
                    and abs((point[1] + point[3]) / 2 - (item[1] + item[3]) / 2) < 0.24
                    for item in group
                )
            ),
            None,
        )
        if match is not None:
            match.append(point)
        else:
            groups.append([point])
    output: List[RegionEvidence] = []
    for group in groups:
        text = " ".join(item[4] for item in group)
        front_signal = len(re.findall(r"PAY|CHECK|CASHIER|ORDER|\$\s*\d|DOLLARS", text, re.I))
        if len(group) < 5 or front_signal < 1:
            continue
        box = (
            max(0.0, min(item[0] for item in group) - 0.04),
            max(0.0, min(item[1] for item in group) - 0.08),
            min(1.0, max(item[2] for item in group) + 0.04),
            min(1.0, max(item[3] for item in group) + 0.08),
        )
        output.append(
            RegionEvidence.create(
                page_number=page_number,
                image=image.crop((int(box[0] * image.width), int(box[1] * image.height), int(box[2] * image.width), int(box[3] * image.height))),
                bbox=box,
                source="ocr_2d_cluster",
                ocr_text=text,
                orientation=90 if (box[3] - box[1]) > (box[2] - box[0]) else 0,
                confidence=min(0.98, 0.55 + front_signal * 0.05),
            )
        )
    return output[:max_regions]


def assess_image_quality(image: Image.Image) -> dict:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    dark_ratio = float((gray < 45).mean())
    bright_ratio = float((gray > 245).mean())
    return {
        "blur_score": round(blur, 2),
        "contrast_score": round(contrast, 2),
        "dark_ratio": round(dark_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
        "needs_enhancement": blur < 90 or contrast < 32 or dark_ratio > 0.35,
    }


def enhance_for_verification(image: Image.Image, target_width: int = 2400) -> Image.Image:
    """Create a deterministic high-resolution OCR/vision verification view."""
    img = crop_to_content(image).convert("RGB")
    if img.width < target_width:
        ratio = target_width / max(1, img.width)
        img = img.resize((target_width, int(img.height * ratio)), Image.Resampling.LANCZOS)
    array = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(array, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    enhanced = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)
    enhanced = cv2.detailEnhance(enhanced, sigma_s=8, sigma_r=0.15)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))


def crop_to_content(image: Image.Image, margin: int = 30) -> Image.Image:
    """Crop large white borders while preserving page/instrument content.

    This reduces image-token usage on scans where the instrument occupies only the top-left
    quadrant. If no reliable bounding box is found, returns the original image.
    """
    img = image.convert("RGB")
    # Difference against a white canvas works well for scanned PDFs.
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    gray = ImageOps.grayscale(diff)
    # Ignore very light paper noise.
    mask = gray.point(lambda p: 255 if p > 18 else 0)
    bbox = mask.getbbox()
    if not bbox:
        return img
    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(img.width, right + margin)
    bottom = min(img.height, bottom + margin)
    # Avoid over-cropping tiny/noisy boxes.
    if (right - left) < img.width * 0.15 or (bottom - top) < img.height * 0.10:
        return img
    return img.crop((left, top, right, bottom))


def maybe_rotate_for_reading(image: Image.Image, angle: float | None) -> Image.Image:
    """Return a page image normalized for reading, using Azure DI page angle.

    Azure Document Intelligence exposes a page-level ``angle`` when the scanned
    text is rotated. In these money-order batches, several real *front* pages are
    scanned upside down. GPT vision can often read them, but it regularly drops
    amount words/cents on inverted money orders, e.g. 442.00 -> 400.00 or
    525.50 -> 525.00. Normalize only when Azure reports a clear right-angle
    rotation so already-upright pages are not damaged.
    """
    if angle is None:
        return image
    try:
        rounded = int(round(float(angle))) % 360
    except (TypeError, ValueError):
        return image

    # DI angles are the observed page orientation. Rotate in the opposite direction
    # for 90/270, and 180 for upside-down pages.
    if rounded in range(80, 101):
        return image.rotate(-90, expand=True)
    if rounded in range(170, 191):
        return image.rotate(180, expand=True)
    if rounded in range(260, 281):
        return image.rotate(90, expand=True)
    return image


def overlapping_reading_regions(
    image: Image.Image,
    *,
    target_long_side_ratio: float = 0.70,
    overlap_ratio: float = 0.12,
    max_regions: int = 6,
) -> List[Image.Image]:
    """Split a dense scan into overlapping readable regions.

    Full-page downscaling makes small, stacked instruments unreadable. Regions
    preserve source pixels and overlap enough that an instrument crossing a
    boundary remains complete in at least one crop.
    """
    img = crop_to_content(image)
    width, height = img.size
    long_side = max(width, height)
    short_side = max(1, min(width, height))
    count = min(max_regions, max(1, int(math.ceil(long_side / (short_side * target_long_side_ratio)))))
    if count <= 1:
        return [img]

    vertical = height >= width
    length = height if vertical else width
    window = min(length, int(length / count * (1.0 + overlap_ratio * 2)))
    step = max(1, int((length - window) / max(count - 1, 1)))
    regions: List[Image.Image] = []
    for index in range(count):
        start = min(index * step, length - window)
        end = min(length, start + window)
        box = (0, start, width, end) if vertical else (start, 0, end, height)
        regions.append(crop_to_content(img.crop(box), margin=12))
    return regions


def recovery_document_regions(image: Image.Image, *, max_regions: int = 12) -> List[Image.Image]:
    """Find readable document candidates across right-angle orientations.

    This is intentionally a failure-recovery detector. It combines conservative
    rectangle geometry, whitespace-separated content blocks, and overlapping
    crops because real scan pages may contain rotated documents with missing
    borders. Equivalent crops from different orientations are deduplicated by
    perceptual image hash.
    """
    candidates: List[Image.Image] = []
    per_orientation = max(2, int(math.ceil(max_regions / 4)))
    for degrees in (0, 90, 180, 270):
        oriented = image.rotate(degrees, expand=True) if degrees else image
        oriented = crop_to_content(oriented)
        regions = instrument_rectangle_regions(oriented, max_regions=per_orientation)
        regions.extend(component_document_regions(oriented, max_regions=per_orientation))
        regions.extend(content_block_regions(oriented, max_regions=per_orientation))
        if not regions:
            regions = overlapping_reading_regions(
                oriented,
                target_long_side_ratio=0.48,
                overlap_ratio=0.20,
                max_regions=per_orientation,
            )
        candidates.extend(regions)

    output: List[Image.Image] = []
    hashes: List[np.ndarray] = []
    for candidate in candidates:
        normalized = crop_to_content(candidate, margin=16)
        thumb = ImageOps.grayscale(normalized).resize((16, 16), Image.Resampling.BILINEAR)
        pixels = np.asarray(thumb, dtype=np.float32)
        digest = pixels > pixels.mean()
        if any(float(np.mean(digest != previous)) < 0.08 for previous in hashes):
            continue
        hashes.append(digest)
        output.append(normalized)
        if len(output) >= max_regions:
            break
    return output


def component_document_regions(image: Image.Image, *, max_regions: int = 12) -> List[Image.Image]:
    """Group dense two-dimensional content into document-shaped regions.

    Unlike whitespace bands, this separates side-by-side documents. Morphology
    joins nearby text, handwriting, amount boxes, and MICR rows inside each
    document while leaving meaningful horizontal and vertical gaps intact.
    """
    img = crop_to_content(image)
    gray = np.asarray(ImageOps.grayscale(img))
    ink = (gray < 225).astype("uint8") * 255
    height, width = ink.shape
    # First join characters into text lines, then nearby lines into documents.
    line_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(9, int(width * 0.018)), max(2, int(height * 0.002)))
    )
    block_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(9, int(width * 0.012)), max(15, int(height * 0.055)))
    )
    joined = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, line_kernel, iterations=2)
    joined = cv2.morphologyEx(joined, cv2.MORPH_CLOSE, block_kernel, iterations=2)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[Tuple[float, Image.Image]] = []
    page_area = float(width * height)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / page_area
        aspect = max(w, h) / max(1, min(w, h))
        if not (0.025 <= area_ratio <= 0.62 and 1.05 <= aspect <= 6.5):
            continue
        margin_x, margin_y = int(w * 0.08), int(h * 0.16)
        box = (
            max(0, x - margin_x),
            max(0, y - margin_y),
            min(width, x + w + margin_x),
            min(height, y + h + margin_y),
        )
        crop = crop_to_content(img.crop(box), margin=16)
        if crop.height > crop.width:
            crop = crop.rotate(90, expand=True)
        regions.append((area_ratio, crop))
    return [crop for _, crop in sorted(regions, key=lambda item: item[0], reverse=True)[:max_regions]]


def ocr_anchor_instrument_regions(
    image: Image.Image,
    words: list,
    *,
    page_number: int,
    max_regions: int = 8,
) -> List[RegionEvidence]:
    """Construct stable front-document crops around repeated OCR anchor groups."""
    anchors: List[Tuple[float, float, str]] = []
    for word in words or []:
        content = str(getattr(word, "content", "") or "")
        if not re.search(r"PAY|EXACTLY|ORDER|MONEY|CASHIER|REMITTER|PURCHASER", content, re.I):
            continue
        xy = _word_xy(word)
        if xy is None:
            continue
        xs, ys = xy
        anchors.append((sum(xs) / len(xs), sum(ys) / len(ys), content))
    if len(anchors) < 4:
        return []

    anchors.sort(key=lambda item: item[1])
    groups: List[List[Tuple[float, float, str]]] = []
    for anchor in anchors:
        if not groups or anchor[1] - groups[-1][-1][1] > 0.16:
            groups.append([anchor])
        else:
            groups[-1].append(anchor)
    groups = [group for group in groups if len(group) >= 2]
    if len(groups) < 2:
        return []

    boundaries: List[Tuple[float, float]] = []
    centers = [sum(item[1] for item in group) / len(group) for group in groups]
    for index, center in enumerate(centers):
        top = 0.0 if index == 0 else (centers[index - 1] + center) / 2
        bottom = 1.0 if index == len(centers) - 1 else (center + centers[index + 1]) / 2
        boundaries.append((max(0.0, top - 0.04), min(1.0, bottom + 0.04)))

    output: List[RegionEvidence] = []
    for group, (top, bottom) in zip(groups[:max_regions], boundaries[:max_regions]):
        crop = image.crop((0, int(top * image.height), image.width, int(bottom * image.height)))
        text = " ".join(item[2] for item in group)
        output.append(
            RegionEvidence.create(
                page_number=page_number,
                image=crop_to_content(crop, margin=16),
                bbox=(0.0, top, 1.0, bottom),
                source="ocr_anchor_group",
                ocr_text=text,
            )
        )
    return output


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = points.astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _warp_quad(image: np.ndarray, quad: np.ndarray) -> Image.Image:
    top_left, top_right, bottom_right, bottom_left = _order_quad(quad)
    width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32"),
        target,
    )
    warped = cv2.warpPerspective(image, matrix, (width, height), borderValue=(255, 255, 255))
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def instrument_rectangle_regions(
    image: Image.Image,
    *,
    max_regions: int = 12,
) -> List[Image.Image]:
    """Detect complete rotated payment-instrument rectangles conservatively."""
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    page_area = float(width * height)
    scale = min(1.0, 1800.0 / max(width, height))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Tuple[float, np.ndarray]] = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (_, _), (rect_width, rect_height), _ = rect
        if rect_width <= 0 or rect_height <= 0:
            continue
        long_side, short_side = max(rect_width, rect_height), min(rect_width, rect_height)
        area_ratio = (long_side * short_side) / (small.shape[0] * small.shape[1])
        aspect = long_side / short_side
        fill = cv2.contourArea(contour) / max(long_side * short_side, 1.0)
        if not (0.035 <= area_ratio <= 0.55 and 1.45 <= aspect <= 4.8 and fill >= 0.08):
            continue
        quad = cv2.boxPoints(rect) / scale
        candidates.append((area_ratio * fill, quad))

    # Long horizontal borders remain detectable when adjacent instruments merge
    # into one contour. Pair nearby line bands into additional full-width crops.
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(50, int(small.shape[1] * 0.18)),
        minLineLength=int(small.shape[1] * 0.42),
        maxLineGap=int(small.shape[1] * 0.08),
    )
    horizontal_y: List[int] = []
    for line in lines[:, 0] if lines is not None else []:
        x1, y1, x2, y2 = line
        if abs(y2 - y1) <= max(5, int(abs(x2 - x1) * 0.04)):
            y = int((y1 + y2) / 2)
            if not horizontal_y or min(abs(y - existing) for existing in horizontal_y) > small.shape[0] * 0.025:
                horizontal_y.append(y)
    horizontal_y.sort()
    min_band = small.shape[0] * 0.09
    max_band = small.shape[0] * 0.34
    for top, bottom in zip(horizontal_y, horizontal_y[1:]):
        band_height = bottom - top
        if not min_band <= band_height <= max_band:
            continue
        margin_y = int(band_height * 0.10)
        quad = np.array(
            [
                [0, max(0, top - margin_y)],
                [small.shape[1] - 1, max(0, top - margin_y)],
                [small.shape[1] - 1, min(small.shape[0] - 1, bottom + margin_y)],
                [0, min(small.shape[0] - 1, bottom + margin_y)],
            ],
            dtype="float32",
        ) / scale
        candidates.append((0.04, quad))

    output: List[Image.Image] = []
    kept_boxes: List[Tuple[float, float, float, float]] = []
    for _, quad in sorted(candidates, key=lambda item: item[0], reverse=True):
        xs, ys = quad[:, 0], quad[:, 1]
        box = (max(0.0, xs.min()), max(0.0, ys.min()), min(width, xs.max()), min(height, ys.max()))
        if any(
            max(0.0, min(box[2], kept[2]) - max(box[0], kept[0]))
            * max(0.0, min(box[3], kept[3]) - max(box[1], kept[1]))
            > 0.65 * min((box[2] - box[0]) * (box[3] - box[1]), (kept[2] - kept[0]) * (kept[3] - kept[1]))
            for kept in kept_boxes
        ):
            continue
        region = _warp_quad(bgr, quad)
        if region.width * region.height < page_area * 0.025:
            continue
        kept_boxes.append(box)
        output.append(crop_to_content(region, margin=16))
        if len(output) >= max_regions:
            break
    return output


def content_block_regions(
    image: Image.Image,
    *,
    max_regions: int = 12,
) -> List[Image.Image]:
    """Split stacked documents at whitespace valleys without assuming a row count."""
    img = crop_to_content(image)
    gray = np.asarray(ImageOps.grayscale(img))
    ink = gray < 235
    row_density = ink.mean(axis=1)
    window = max(5, int(img.height * 0.008))
    smooth = np.convolve(row_density, np.ones(window) / window, mode="same")
    active = (smooth > max(0.0015, float(np.percentile(smooth[smooth > 0], 20)) if np.any(smooth > 0) else 0.0015)).astype("uint8")
    # Connect nearby printed rows inside one instrument while preserving the
    # substantially larger whitespace gaps between stacked instruments.
    join = max(9, int(img.height * 0.035))
    active = cv2.morphologyEx(active.reshape(-1, 1), cv2.MORPH_CLOSE, np.ones((join, 1), dtype="uint8")).reshape(-1) > 0

    groups: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= img.height * 0.07:
                groups.append((start, index))
            start = None
    if start is not None and img.height - start >= img.height * 0.07:
        groups.append((start, img.height))

    output: List[Image.Image] = []
    for top, bottom in groups[:max_regions]:
        band = img.crop((0, max(0, top - window * 2), img.width, min(img.height, bottom + window * 2)))
        region = crop_to_content(band, margin=16)
        aspect = max(region.width, region.height) / max(1, min(region.width, region.height))
        area_ratio = region.width * region.height / max(1, img.width * img.height)
        if 1.35 <= aspect <= 5.5 and 0.035 <= area_ratio <= 0.55:
            output.append(region if region.width >= region.height else region.rotate(90, expand=True))
    return output


from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from PIL import Image, ImageOps

from money_order_validator.validation import is_aba_routing_number


FRONT_RE = re.compile(r"PAY|ORDER|EXACTLY|MONEY|CASHIER|CHECK|REMITTER|\$\s*[\d,]+", re.I)
BACK_RE = re.compile(r"ENDORSE|DEPOSITORY|SERVICE\s+CHARGE|AGREEMENT|DO\s+NOT\s+WRITE|FOR\s+DEPOSIT", re.I)


@dataclass(frozen=True)
class Anchor:
    kind: str
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


@dataclass
class Proposal:
    bbox: tuple[float, float, float, float]
    sources: set[str] = field(default_factory=set)
    anchors: List[Anchor] = field(default_factory=list)
    score: float = 0.0
    orientation: int = 0
    confidence: float = 0.0
    review_flags: List[str] = field(default_factory=list)


@dataclass
class PageFrontCensus:
    classification: str
    visible_front_count: int
    confidence: float
    regions: List[RegionEvidence] = field(default_factory=list)
    signals: dict = field(default_factory=dict)



def extract_anchors(words: Iterable[object]) -> List[Anchor]:
    anchors: List[Anchor] = []
    for word in words or []:
        text = str(getattr(word, "content", "") or "")
        xy = _word_xy(word)
        if xy is None:
            continue
        xs, ys = xy
        upper, digits = text.upper(), re.sub(r"\D", "", text)
        kind: Optional[str] = None
        if BACK_RE.search(upper):
            kind = "back"
        elif re.search(r"PAY|ORDER|EXACTLY|MONEY|CASHIER|REMITTER", upper):
            kind = "front"
        elif re.search(r"\$\s*[\d,]+(?:\.\d{2})?", text) or re.fullmatch(r"[\d,]+\.\d{2}", text):
            kind = "amount"
        elif len(digits) >= 4:
            kind = "micr_or_number"
        if kind:
            anchors.append(Anchor(kind, text, (min(xs), min(ys), max(xs), max(ys)), float(getattr(word, "confidence", 0.0) or 0.0)))
    return anchors


def _territory_proposals(anchors: List[Anchor]) -> List[Proposal]:
    seeds = [anchor for anchor in anchors if anchor.kind in {"micr_or_number", "front", "amount"}]
    proposals: List[Proposal] = []
    for seed in seeds:
        sx, sy = seed.center
        nearby = [
            anchor for anchor in anchors
            if abs(anchor.center[0] - sx) <= 0.30 and abs(anchor.center[1] - sy) <= 0.24
        ]
        kinds = {anchor.kind for anchor in nearby}
        if "back" in kinds and not ({"amount", "front"} <= kinds):
            continue
        if not ("amount" in kinds and kinds & {"front", "micr_or_number"}):
            continue
        box = (
            max(0.0, min(anchor.bbox[0] for anchor in nearby) - 0.07),
            max(0.0, min(anchor.bbox[1] for anchor in nearby) - 0.12),
            min(1.0, max(anchor.bbox[2] for anchor in nearby) + 0.07),
            min(1.0, max(anchor.bbox[3] for anchor in nearby) + 0.12),
        )
        proposals.append(Proposal(box, {"anchor_territory"}, nearby))
    return proposals


def _ink_proposals(image: Image.Image) -> List[Proposal]:
    gray = np.asarray(ImageOps.grayscale(image))
    ink = (gray < 225).astype("uint8") * 255
    height, width = ink.shape
    joined = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(width * 0.018)), max(15, int(height * 0.05)))),
        iterations=2,
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output: List[Proposal] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = (w * h) / float(width * height)
        aspect = max(w, h) / max(1, min(w, h))
        if 0.025 <= area <= 0.70 and 1.05 <= aspect <= 7.0:
            mx, my = int(w * 0.08), int(h * 0.14)
            output.append(
                Proposal(
                    (max(0, x - mx) / width, max(0, y - my) / height, min(width, x + w + mx) / width, min(height, y + h + my) / height),
                    {"ink_component"},
                )
            )
    return output


def _consensus(proposals: List[Proposal], anchors: List[Anchor]) -> List[Proposal]:
    groups: List[List[Proposal]] = []
    for proposal in proposals:
        match = next((group for group in groups if any(_bbox_iou(proposal.bbox, item.bbox) >= 0.52 for item in group)), None)
        if match is None:
            groups.append([proposal])
        else:
            match.append(proposal)
    output: List[Proposal] = []
    for group in groups:
        best = max(group, key=lambda item: len(item.sources))
        sources = set().union(*(item.sources for item in group))
        local = [anchor for anchor in anchors if best.bbox[0] <= anchor.center[0] <= best.bbox[2] and best.bbox[1] <= anchor.center[1] <= best.bbox[3]]
        kinds = {anchor.kind for anchor in local}
        score = 2 * len(sources) + 5 * ("micr_or_number" in kinds) + 4 * ("front" in kinds) + 4 * ("amount" in kinds) - 5 * ("back" in kinds)
        micr_count = sum(anchor.kind == "micr_or_number" for anchor in local)
        flags: List[str] = []
        if micr_count > 3:
            score -= 5
            flags.append("multiple_identity_clusters")
        if "back" in kinds:
            flags.append("back_zone_overlap")
        best.sources, best.anchors, best.score, best.review_flags = sources, local, float(score), flags
        best.orientation = 90 if (best.bbox[3] - best.bbox[1]) > (best.bbox[2] - best.bbox[0]) else 0
        best.confidence = min(0.99, max(0.0, score / 15.0))
        if score >= 5 and "amount" in kinds and kinds & {"front", "micr_or_number"}:
            output.append(best)
    return sorted(output, key=lambda item: item.score, reverse=True)


def _transform_bbox(
    bbox: tuple[float, float, float, float], degrees: int
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    if degrees == 90:
        return (y0, 1.0 - x1, y1, 1.0 - x0)
    if degrees == 180:
        return (1.0 - x1, 1.0 - y1, 1.0 - x0, 1.0 - y0)
    if degrees == 270:
        return (1.0 - y1, x0, 1.0 - y0, x1)
    return bbox


def rotation_sweep_census(image: Image.Image, page_number: int, max_regions: int = 12) -> List[RegionEvidence]:
    """Find geometry proposals at every right angle and map them to PDF coordinates."""

    output: List[RegionEvidence] = []
    for degrees in (0, 90, 180, 270):
        oriented = image.rotate(degrees, expand=True) if degrees else image
        for region in document_region_proposals(oriented, page_number=page_number, max_regions=max_regions):
            original_bbox = _transform_bbox(region.bbox, (360 - degrees) % 360)
            x0, y0, x1, y1 = original_bbox
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                continue
            recovered = RegionEvidence.create(
                page_number=page_number,
                image=image.crop((int(x0 * image.width), int(y0 * image.height), int(x1 * image.width), int(y1 * image.height))),
                bbox=original_bbox,
                source="rotation_census",
                orientation=(360 - degrees) % 360,
                confidence=max(0.35, region.confidence - 0.05),
            )
            recovered.evidence["rotation_census_degrees"] = degrees
            output.append(recovered)
    return merge_region_proposals(output, iou_threshold=0.58)[:max_regions]


def forced_territory_partition(
    image: Image.Image,
    page_number: int,
    expected_regions: int,
    max_regions: int = 12,
) -> List[RegionEvidence]:
    """Partition low-yield pages only when controls prove multiple regions are missing."""
    if expected_regions <= 1:
        return []
    gray = np.asarray(ImageOps.grayscale(image))
    ink = gray < 225
    height, width = ink.shape
    vertical_profile = ink.mean(axis=0)
    horizontal_profile = ink.mean(axis=1)
    axis = "y" if height >= width else "x"
    profile = horizontal_profile if axis == "y" else vertical_profile
    smooth_width = max(3, int(profile.size * 0.015))
    smooth = np.convolve(profile, np.ones(smooth_width) / smooth_width, mode="same")
    candidates = sorted(range(int(profile.size * 0.08), int(profile.size * 0.92)), key=lambda i: smooth[i])
    cuts: List[int] = []
    min_gap = profile.size / max(2, expected_regions * 2)
    for candidate in candidates:
        if all(abs(candidate - cut) >= min_gap for cut in cuts):
            cuts.append(candidate)
        if len(cuts) >= expected_regions - 1:
            break
    cuts = sorted(cuts)
    bounds = [0, *cuts, profile.size]
    output: List[RegionEvidence] = []
    for start, end in zip(bounds, bounds[1:]):
        if end - start < profile.size * 0.08:
            continue
        bbox = (0.0, start / height, 1.0, end / height) if axis == "y" else (start / width, 0.0, end / width, 1.0)
        x0, y0, x1, y1 = bbox
        region = RegionEvidence.create(
            page_number=page_number,
            image=image.crop((int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))),
            bbox=bbox,
            source="forced_territory_partition",
            orientation=90 if axis == "x" else 0,
            confidence=0.32,
        )
        region.evidence.update({"forced_k": expected_regions, "partition_axis": axis})
        output.append(region)
    return output[:max_regions]


def topology_check_regions(
    image: Image.Image,
    page_number: int,
    *,
    words: Iterable[object] = (),
    page_kind: Optional[str] = None,
    front_score: int = 0,
    max_regions: int = 8,
) -> List[RegionEvidence]:
    """Detect check-like regions from dense horizontal topology without OCR."""
    gray = np.asarray(ImageOps.grayscale(image))
    height, width = gray.shape
    ink = gray < 225
    # Join text and ruled lines inside checks while leaving sparse endorsement
    # islands disconnected from the dominant instrument band.
    mask = ink.astype("uint8") * 255
    joined = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(width * 0.012)), max(3, int(height * 0.006)))),
        iterations=2,
    )
    row_profile = (joined > 0).mean(axis=1)
    active = row_profile >= max(0.012, float(np.percentile(row_profile[row_profile > 0], 55)) * 0.20) if np.any(row_profile > 0) else np.zeros(height, dtype=bool)
    groups: List[tuple[int, int]] = []
    start: Optional[int] = None
    gap = max(4, int(height * 0.035))
    last = -gap
    for index, value in enumerate(active):
        if value:
            if start is None or index - last > gap:
                if start is not None:
                    groups.append((start, last + 1))
                start = index
            last = index
    if start is not None:
        groups.append((start, last + 1))
    bands = [
        (top, bottom)
        for top, bottom in groups
        if bottom - top >= height * 0.12
        and np.mean(row_profile[top:bottom]) >= 0.015
    ]
    output: List[RegionEvidence] = []
    anchors = extract_anchors(words)
    page_text = " ".join(str(getattr(word, "content", "") or "") for word in words).upper()
    # Back-dominant pages may still contain visible fronts beside endorsement
    # stamps. Deposit/report/receipt pages remain control-only unless their
    # front score independently proves mixed content.
    control_only_kinds = {"deposit_slip", "deposit_report", "receipt", "batch_header", "blank"}
    if page_kind in control_only_kinds and front_score < 6:
        return []
    if re.search(r"BATCH\s+DETAILS|ACTUAL\s+ITEMS|BATCH\s+AMOUNT|LINE\s+ITEMS", page_text):
        return []
    for top, bottom in sorted(bands, key=lambda pair: np.mean(row_profile[pair[0]:pair[1]]), reverse=True):
        margin_y = int((bottom - top) * 0.08)
        top, bottom = max(0, top - margin_y), min(height, bottom + margin_y)
        local = ink[top:bottom]
        column_profile = local.mean(axis=0)
        smooth_width = max(5, int(width * 0.02))
        smooth = np.convolve(column_profile, np.ones(smooth_width) / smooth_width, mode="same")
        center_start, center_end = int(width * 0.30), int(width * 0.70)
        valley = int(np.argmin(smooth[center_start:center_end])) + center_start
        baseline = float(np.percentile(smooth[smooth > 0], 55)) if np.any(smooth > 0) else 0.0
        cuts = [0, valley, width] if baseline and smooth[valley] <= baseline * 0.45 else [0, width]
        for left, right in zip(cuts, cuts[1:]):
            if right - left < width * 0.22:
                continue
            bbox = (left / width, top / height, right / width, bottom / height)
            region_width = bbox[2] - bbox[0]
            # Dense topology often begins at the amount/payee lines and misses
            # the sparse payer/header area. Expand upward from the detected
            # bottom using a broad personal-check aspect prior.
            resolved_bottom = max(bbox[1], bbox[3] - region_width * 0.11)
            target_height = region_width / 1.45
            bbox = (bbox[0], max(0.0, resolved_bottom - target_height), bbox[2], resolved_bottom)
            region_height = bbox[3] - bbox[1]
            aspect = region_width / max(region_height, 0.001)
            if not (0.32 <= region_width <= 0.68 and 0.16 <= region_height <= 0.48 and 1.35 <= aspect <= 3.25):
                continue
            local = [
                anchor
                for anchor in anchors
                if bbox[0] <= anchor.center[0] <= bbox[2]
                and bbox[1] <= anchor.center[1] <= bbox[3]
            ]
            kinds = {anchor.kind for anchor in local}
            # When OCR evidence exists, require front/amount evidence and reject
            # back-only bands. Empty OCR remains a geometry-only shadow proposal.
            if anchors and "front" not in kinds:
                continue
            if kinds == {"back"}:
                continue
            region = RegionEvidence.create(
                page_number=page_number,
                image=image.crop((left, top, right, bottom)),
                bbox=bbox,
                source="topology_check_band",
                ocr_text="\n".join(
                    str(getattr(word, "content", "") or "")
                    for word in words
                    if (
                        (polygon := tuple(getattr(word, "polygon", ()) or ()))
                        and len(polygon) >= 4
                        and bbox[0] <= sum(polygon[0::2]) / len(polygon[0::2]) <= bbox[2]
                        and bbox[1] <= sum(polygon[1::2]) / len(polygon[1::2]) <= bbox[3]
                    )
                ),
                confidence=0.68,
            )
            region.evidence.update(
                {
                    "vertical_split": len(cuts) == 3,
                    "row_density": float(np.mean(row_profile[top:bottom])),
                    "anchor_kinds": sorted(kinds),
                }
            )
            output.append(region)
        if len(output) >= max_regions:
            break
    return output[:max_regions]


def segment_page(image: Image.Image, words: Iterable[object], page_number: int, max_regions: int = 12) -> List[RegionEvidence]:
    """Return zero-token, self-verified multi-signal instrument proposals."""
    anchors = extract_anchors(words)
    proposals = _consensus([*_territory_proposals(anchors), *_ink_proposals(image)], anchors)
    output: List[RegionEvidence] = []
    for proposal in proposals[:max_regions]:
        x0, y0, x1, y1 = proposal.bbox
        output.append(
            RegionEvidence.create(
                page_number=page_number,
                image=image.crop((int(x0 * image.width), int(y0 * image.height), int(x1 * image.width), int(y1 * image.height))),
                bbox=proposal.bbox,
                source="segmentation_consensus",
                ocr_text="\n".join(f"{anchor.kind}: {anchor.text}" for anchor in proposal.anchors),
                orientation=proposal.orientation,
                confidence=proposal.confidence,
            )
        )
    return output


def page_front_census(
    image: Image.Image,
    words: Iterable[object],
    page_number: int,
    *,
    page_kind: str,
    front_score: int,
    max_regions: int = 12,
) -> PageFrontCensus:
    """Classify visible-front topology before field extraction."""

    words = list(words or [])
    topology = topology_check_regions(
        image,
        page_number,
        words=words,
        page_kind=page_kind,
        front_score=front_score,
        max_regions=max_regions,
    )
    graph = evidence_graph_region_proposals(
        image, words, page_number=page_number, max_regions=max_regions
    )
    clusters = ocr_cluster_region_proposals(
        image, words, page_number=page_number, max_regions=max_regions
    )
    consensus = segment_page(image, words, page_number, max_regions=max_regions)
    candidates = merge_region_proposals(
        [*topology, *graph, *clusters, *consensus],
        iou_threshold=0.58,
    )[:max_regions]
    strong = [
        region
        for region in candidates
        if region.confidence >= 0.55
        and (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1]) >= 0.045
    ]
    count = len(strong)
    control_kind = page_kind in {"deposit_slip", "deposit_report", "receipt", "batch_header", "report_with_instruments"}
    if count >= 2:
        classification = "report_with_fronts" if control_kind else "multi_front"
    elif count == 1:
        classification = "report_with_fronts" if control_kind else "single_front"
    elif page_kind == "back_page":
        classification = "back_only"
    else:
        classification = "uncertain"
    independent_sources = {
        region.source.split("_valley_split")[0]
        for region in strong
    }
    confidence = min(0.99, 0.45 + count * 0.12 + min(3, len(independent_sources)) * 0.08)
    return PageFrontCensus(
        classification=classification,
        visible_front_count=count,
        confidence=confidence,
        regions=strong,
        signals={
            "topology": len(topology),
            "evidence_graph": len(graph),
            "ocr_clusters": len(clusters),
            "consensus": len(consensus),
            "independent_sources": sorted(independent_sources),
        },
    )


def verify_recovered_instrument(row: dict, region: RegionEvidence) -> tuple[bool, List[str]]:
    """Apply the zero-token exit gate to one isolated recovery result."""
    flags: List[str] = []
    serial = re.sub(r"\D", "", str(row.get("serial_number") or ""))
    micr = re.sub(r"\D", "", str(row.get("micr_line") or ""))
    has_identity = bool(serial or micr)
    has_amount = row.get("amount_numeric") is not None or row.get("amount_candidate") is not None

    if region.confidence < 0.30 and region.source != "full_page_recovery":
        flags.append("low_region_confidence")
    if not has_identity:
        flags.append("missing_document_identity")
    if serial and is_aba_routing_number(serial):
        flags.append("routing_number_as_identity")
    if not has_amount:
        flags.append("missing_amount_evidence")
    if row.get("amount_status") == "conflict":
        flags.append("amount_evidence_conflict")
    if row.get("image_quality") == "unclear":
        flags.append("unclear_instrument_image")

    blocking = {
        "missing_document_identity",
        "routing_number_as_identity",
        "missing_amount_evidence",
        "amount_evidence_conflict",
    }
    return not bool(blocking.intersection(flags)), flags
