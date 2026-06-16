from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Security, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, RedirectResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from datetime import datetime
import os
import json
import logging
import uvicorn
import traceback
import asyncio
from typing import Optional, Tuple, Dict, Any, List
import re
import uuid

# Import all extractors
from proof_of_income import extract_employment_data_ocr
from id_document import extract_id_data
from insurance import extract_insurance_data_ocr
from invoice import extract_invoice_data
from bulk_invoice import extract_invoice_data as extract_bulk_invoice_data
from vendor import extract_vendor_data_ocr

# Import chatbot services
from chatbot_services.azure_search_service import AzureSearchService
from chatbot_services.llm_service import LLMService
from chatbot_services.document_intelligence_service import DocumentIntelligenceService
from chatbot_services.redis_service import get_redis_client, close_redis
from chatbot_services.blob_service import BlobService
import chatbot_config

# Import money order validator service
from money_order_validator import process_file as validate_checks_file

class PersistenceService:
    """No-op stub — chat history persistence is disabled."""
    async def initialize(self): pass
    async def save_chat_exchange(self, **kwargs): pass
    async def ensure_session(self, session_id: str): pass
    async def save_upload(self, **kwargs) -> str: return str(uuid.uuid4())
    async def delete_session(self, session_id: str): pass

from pathlib import Path
from dotenv import load_dotenv

# .env lives at services/.env — one directory above this file.
load_dotenv(Path(__file__).resolve().parent.parent / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if not chatbot_config.CHATBOT_API_KEY:
    logger.warning("CHATBOT_API_KEY is not configured; chatbot endpoints requiring API key will be unavailable.")

# Reduce noisy third-party logs
for _noisy_logger in [
    "httpx", "httpcore", "azure",
    "azure.core.pipeline.policies.http_logging_policy",
    "urllib3", "openai", "gunicorn.access", "uvicorn.access",
]:
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# ── Chatbot file validation (magic bytes) ───────────────────────────────────
CHATBOT_ALLOWED_SIGNATURES = [
    b'%PDF',
    b'\xff\xd8\xff',
    b'\x89PNG\r\n\x1a\n',
    b'II*\x00',
    b'MM\x00*',
    b'BM',
    b'PK\x03\x04',
]

CHATBOT_ALLOWED_CONTENT_TYPES = [
    'application/pdf',
    'image/jpeg',
    'image/jpg',
    'image/png',
    'image/tiff',
    'image/bmp',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain',
]


def validate_chatbot_file_content(content: bytes, content_type: str) -> bool:
    """Validate uploaded file bytes against expected signatures for the chatbot upload endpoint."""
    for sig in CHATBOT_ALLOWED_SIGNATURES:
        if content[:len(sig)] == sig:
            return True
    if content_type == 'text/plain':
        try:
            content[:1024].decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False
    return False


# ── Chatbot rate limiting helpers ────────────────────────────────────────────
RATE_WINDOW_SECONDS = {
    "second": 1, "seconds": 1,
    "minute": 60, "minutes": 60,
    "hour": 3600, "hours": 3600,
    "day": 86400, "days": 86400,
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


# ── Chatbot service singletons ───────────────────────────────────────────────
search_service = AzureSearchService()
llm_service = LLMService()
chatbot_doc_intelligence_service = DocumentIntelligenceService()
blob_service = BlobService()
persistence_service = PersistenceService()


# ── App lifespan (startup / shutdown) ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await persistence_service.initialize()
    yield
    await close_redis()


app = FastAPI(title="Yotta AI Services", version="1.0.0", lifespan=lifespan)

# Configuration - Different API keys for different document types
API_KEY_INVOICE     = os.getenv("API_KEY_INVOICE", "")
API_KEY_POI         = os.getenv("API_KEY_POI", "")
API_KEY_ID          = os.getenv("API_KEY_ID", "")
API_KEY_INSURANCE   = os.getenv("API_KEY_INSURANCE", "")
API_KEY_BULKINVOICE = os.getenv("API_KEY_BULKINVOICE", "")
API_KEY_VENDOR      = os.getenv("API_KEY_VENDOR", "")
API_KEY_CHECKVALIDATION = os.getenv("API_KEY_CHECKVALIDATION", "")
API_KEY_MASTER      = os.getenv("API_KEY_MASTER", "")  # Master key for all services
ENABLE_FRONTEND     = os.getenv("ENABLE_FRONTEND", "true").lower() == "true"

# File validation constants
MAX_FILE_SIZE      = 15 * 1024 * 1024  # 15MB
BULK_MAX_FILE_SIZE = 5  * 1024 * 1024  # 5MB per file for bulk invoice
ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.heic', '.heif'}

# Semaphore for bulk invoice — limit concurrent GPT calls
_gpt_semaphore = asyncio.Semaphore(3)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Private-Network"] = "true"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

app.add_middleware(PrivateNetworkAccessMiddleware)

# ==================== CHATBOT API KEY VERIFICATION ====================

chatbot_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_chatbot_api_key(api_key: str = Security(chatbot_api_key_header)):
    """Verify API key for chatbot endpoints."""
    if api_key != chatbot_config.CHATBOT_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return True


# ==================== CHATBOT PYDANTIC MODELS ====================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[dict]
    session_id: str


class CleanupRequest(BaseModel):
    session_id: str


# ==================== EXTRACTOR API KEY VERIFICATION ====================

async def verify_api_key_invoice(x_api_key: Optional[str] = Header(None)):
    """Verify API key for invoice extraction"""
    if not API_KEY_INVOICE and not API_KEY_MASTER:
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Include 'X-API-Key' header.")
    
    if x_api_key != API_KEY_INVOICE and x_api_key != API_KEY_MASTER:
        raise HTTPException(status_code=401, detail="Invalid API key for invoice extraction")
    
    return True

async def verify_api_key_poi(x_api_key: Optional[str] = Header(None)):
    """Verify API key for POI extraction"""
    if not API_KEY_POI and not API_KEY_MASTER:
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Include 'X-API-Key' header.")
    
    if x_api_key != API_KEY_POI and x_api_key != API_KEY_MASTER:
        raise HTTPException(status_code=401, detail="Invalid API key for POI extraction")
    
    return True

async def verify_api_key_id(x_api_key: Optional[str] = Header(None)):
    """Verify API key for ID extraction"""
    if not API_KEY_ID and not API_KEY_MASTER:
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Include 'X-API-Key' header.")
    
    if x_api_key != API_KEY_ID and x_api_key != API_KEY_MASTER:
        raise HTTPException(status_code=401, detail="Invalid API key for ID extraction")
    
    return True

async def verify_api_key_insurance(x_api_key: Optional[str] = Header(None)):
    """Verify API key for insurance extraction"""
    if not API_KEY_INSURANCE and not API_KEY_MASTER:
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Include 'X-API-Key' header.")
    
    if x_api_key != API_KEY_INSURANCE and x_api_key != API_KEY_MASTER:
        raise HTTPException(status_code=401, detail="Invalid API key for insurance extraction")
    
    return True

async def verify_api_key_bulkinvoice(x_api_key: Optional[str] = Header(None)):
    """Verify API key for bulk invoice extraction"""
    if not API_KEY_BULKINVOICE and not API_KEY_MASTER:
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Include 'X-API-Key' header.")
    
    if x_api_key != API_KEY_BULKINVOICE and x_api_key != API_KEY_MASTER:
        raise HTTPException(status_code=401, detail="Invalid API key for bulk invoice extraction")
    
    return True

async def verify_api_key_vendor(x_api_key: Optional[str] = Header(None)):
    """Verify API key for vendor extraction"""
    if not API_KEY_VENDOR and not API_KEY_MASTER:
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Include 'X-API-Key' header.")
    
    if x_api_key != API_KEY_VENDOR and x_api_key != API_KEY_MASTER:
        raise HTTPException(status_code=401, detail="Invalid API key for vendor extraction")
    
    return True

async def verify_api_key_checkvalidation(x_api_key: Optional[str] = Header(None)):
    """Verify API key for check validation"""
    if not API_KEY_CHECKVALIDATION and not API_KEY_MASTER:
        return True

    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Include 'X-API-Key' header.")

    if x_api_key != API_KEY_CHECKVALIDATION and x_api_key != API_KEY_MASTER:
        raise HTTPException(status_code=401, detail="Invalid API key for check validation")

    return True

# ==================== HELPER FUNCTIONS ====================

def count_non_null_fields(data: Any) -> int:
    """Recursively count number of fields with actual values"""
    if isinstance(data, dict):
        return sum(count_non_null_fields(v) for v in data.values())
    if isinstance(data, list):
        return sum(count_non_null_fields(item) for item in data)
    return 0 if data in (None, "", []) else 1

# ==================== INVOICE HELPERS ====================

def requires_human_review_invoice(extracted_data) -> Tuple[bool, str]:
    """Determine if invoice needs human review"""
    if not extracted_data:
        return True, "No data extracted"

    invoices = [extracted_data] if isinstance(extracted_data, dict) else extracted_data
    reasons: list[str] = []

    for idx, data in enumerate(invoices, start=1):
        vital_fields = ["invoice_id", "vendor_name", "line_items"]
        missing_vital = [
            field for field in vital_fields
            if data.get(field) in (None, "", [])
        ]
        if missing_vital:
            reasons.append(f"Invoice {idx}: missing vital field(s): {', '.join(missing_vital)}")

    if reasons:
        return True, " ; ".join(reasons)
    return False, ""

# ==================== POI HELPERS ====================

def _value_is_empty(value: Any) -> bool:
    """Check if a value is considered empty"""
    return value in (None, "", [])

def _iter_jobs(extracted_data: Dict[str, Any]):
    """Helper generator to iterate through (applicant_name, job_dict) pairs"""
    applicants = extracted_data.get("applicants", []) or []
    for applicant in applicants:
        applicant_name = applicant.get("applicant")
        jobs = applicant.get("jobs", []) or []
        for job in jobs:
            yield applicant_name, job

def requires_human_review_poi(extracted_data: dict) -> Tuple[bool, str]:
    """Determine if POI needs human review"""
    if not extracted_data:
        return True, "No data extracted"

    applicants = extracted_data.get("applicants", []) or []
    if not applicants:
        return True, "No applicants found"

    has_complete_record = False
    missing_vital_fields: set[str] = set()

    for applicant_name, job in _iter_jobs(extracted_data):
        avg_income = job.get("average_monthly_income")

        missing_for_this = []
        if _value_is_empty(applicant_name):
            missing_for_this.append("applicant")
        # if _value_is_empty(avg_income):
            # missing_for_this.append("average_monthly_income")

        if not missing_for_this:
            has_complete_record = True
        else:
            missing_vital_fields.update(missing_for_this)

    if has_complete_record:
        return False, ""

    if not missing_vital_fields:
        return True, "Missing vital fields"

    missing_str = ", ".join(sorted(missing_vital_fields))
    return True, f"Missing vital field(s): {missing_str}"

def has_proof_of_employment(extracted_data: dict) -> bool:
    """Determine if the document appears to be proof of employment/income"""
    for applicant_name, job in _iter_jobs(extracted_data):
        if not _value_is_empty(applicant_name):
            return True
    return False

def filter_verification_applications(extracted_data: dict) -> dict:
    """Remove jobs that are only from employment verification applications"""
    if not extracted_data or "applicants" not in extracted_data:
        return extracted_data
    
    filtered_applicants = []
    for applicant in extracted_data.get("applicants", []):
        valid_jobs = [
            job for job in applicant.get("jobs", [])
            if job.get("document_type") not in ["employment verification application"]
        ]
        if valid_jobs:
            filtered_applicants.append({
                "applicant": applicant.get("applicant"),
                "jobs": valid_jobs
            })
    
    return {"applicants": filtered_applicants}

def get_vital_fields_status_poi(extracted_data: dict) -> Dict[str, bool]:
    """Return a summary of whether we found any non-empty values for each vital field"""
    has_applicant = False
    for applicant_name, job in _iter_jobs(extracted_data):
        if not _value_is_empty(applicant_name):
            has_applicant = True
    return {"applicant": has_applicant}

PERIODS_PER_YEAR = {
    "weekly": 52,
    "bi-weekly": 26,
    "biweekly": 26,
    "semi-monthly": 24,
    "semimonthly": 24,
    "monthly": 12,
}

def _parse_currency(value: Any) -> Optional[float]:
    """Parse a numeric amount into a float"""
    if not value or not isinstance(value, str):
        return None
    matches = re.findall(r"[\d,]+(?:\.\d+)?", value)
    if not matches:
        return None
    num_str = matches[-1]
    try:
        return float(num_str.replace(",", ""))
    except ValueError:
        return None

def recompute_average_monthly_income(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute average_monthly_income and keep income_calculation_details consistent"""
    if not extracted_data:
        return extracted_data

    for _, job in _iter_jobs(extracted_data):
        yearly_salary_str = job.get("yearly_salary")
        yearly = _parse_currency(yearly_salary_str) if yearly_salary_str else None

        if yearly:
            monthly = yearly / 12.0
            monthly_rounded = round(monthly, 2)
            monthly_str = f"{monthly_rounded:.2f}"
            job["average_monthly_income"] = monthly_str
            job["income_calculation_details"] = (
                f"Based on yearly salary: {yearly_salary_str} / 12 months = {monthly_str}/month"
            )
            continue

        checks = job.get("three_consecutive_checks") or []
        pay_freq_raw = job.get("pay_frequency")
        pay_freq = pay_freq_raw.strip().lower() if isinstance(pay_freq_raw, str) else None
        periods_per_year = PERIODS_PER_YEAR.get(pay_freq) if pay_freq else None

        amounts: List[float] = []
        for entry in checks:
            amt = _parse_currency(entry)
            if amt is not None:
                amounts.append(amt)

        if amounts and periods_per_year:
            if pay_freq == "monthly":
                avg_monthly = sum(amounts) / len(amounts)
                avg_monthly_rounded = round(avg_monthly, 2)
                monthly_str = f"{avg_monthly_rounded:.2f}"
                job["average_monthly_income"] = monthly_str
                job["income_calculation_details"] = (
                    f"Based on {len(amounts)} monthly paychecks: "
                    f"average monthly income = {monthly_str}/month"
                )
            else:
                avg_check = sum(amounts) / len(amounts)
                annual = avg_check * periods_per_year
                monthly = annual / 12.0
                avg_check_rounded = round(avg_check, 2)
                monthly_rounded = round(monthly, 2)
                avg_check_str = f"{avg_check_rounded:.2f}"
                monthly_str = f"{monthly_rounded:.2f}"
                job["average_monthly_income"] = monthly_str
                job["income_calculation_details"] = (
                    f"Based on {len(amounts)} {pay_freq} paychecks: "
                    f"average paycheck = {avg_check_str}. "
                    f"{(pay_freq_raw or pay_freq).capitalize()} pay: "
                    f"{avg_check_str} × {periods_per_year} pay periods / 12 months = {monthly_str}/month"
                )

    return extracted_data

# ==================== ID HELPERS ====================

def filter_valid_ids(ids: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out IDs that have neither full_name nor id_number.
    These are likely noise/artifacts and not actual ID extractions.
    """
    valid_ids = []
    for id_data in ids:
        has_name = id_data.get("full_name") not in (None, "", [])
        has_id_number = id_data.get("id_number") not in (None, "", [])
        
        # Keep ID only if it has at least one of these critical fields
        if has_name or has_id_number:
            valid_ids.append(id_data)
    
    return valid_ids

def count_non_null_fields_id(data: Dict[str, Any]) -> int:
    """Count number of fields with actual values in a single ID dict"""
    if not data:
        return 0
    return sum(1 for value in data.values() if value not in (None, "", []))

def requires_human_review_id(extracted_data) -> Tuple[bool, str]:
    """Determine if ID needs human review"""
    if not extracted_data:
        return True, "No data extracted"

    if isinstance(extracted_data, dict):
        ids = [extracted_data]
    else:
        ids = extracted_data

    # Filter out invalid IDs (missing both name and ID number)
    ids = filter_valid_ids(ids)
    
    # If all IDs were filtered out, return error
    if not ids:
        return True, "No valid ID data found"

    reasons: list[str] = []

    for idx, data in enumerate(ids, start=1):
        doc_type = (data.get("document_type") or "").lower()

        if doc_type == "social_security":
            vital_fields = ["id_number", "full_name"]
            min_total_fields = 2
        else:
            vital_fields = ["id_number", "full_name", "date_of_birth"]
            min_total_fields = 6

        missing_vital = [
            field for field in vital_fields
            if data.get(field) in (None, "", [])
        ]

        if missing_vital:
            missing_str = ", ".join(missing_vital)
            reasons.append(f"ID {idx}: missing vital field(s): {missing_str}")
            continue

        field_count = count_non_null_fields_id(data)
        if field_count < min_total_fields:
            reasons.append(
                f"ID {idx}: insufficient data extracted "
                f"({field_count}/{min_total_fields} required fields)"
            )

    if reasons:
        return True, " ; ".join(reasons)
    return False, ""

# ==================== INSURANCE HELPERS ====================

def requires_human_review_insurance(extracted_data: dict) -> Tuple[bool, str]:
    """Determine if insurance needs human review"""
    if not extracted_data:
        return True, "No data extracted"
    
    vital_fields = ["policy_number", "insured_name", "insurance_company"]
    missing_vital = [
        field for field in vital_fields
        if extracted_data.get(field) in (None, "", [])
    ]
    
    if missing_vital:
        missing_str = ", ".join(missing_vital)
        return True, f"Missing vital field(s): {missing_str}"
    
    field_count = count_non_null_fields(extracted_data)
    if field_count < 6:
        return True, f"Insufficient data extracted ({field_count}/6 required fields)"
    
    return False, ""

# ==================== ROOT ENDPOINT ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    frontend_links = ""
    if ENABLE_FRONTEND:
        frontend_links = """
                <div style="margin-top: 30px;">
                    <h3>Frontend Tools:</h3>
                    <ul style="line-height: 2;">
                        <li><a href="/test" style="color: #0066cc;">🧪 Test Interface</a> - Upload and test documents</li>
                        <li><a href="/chat" style="color: #0066cc;">💬 Chatbot</a> - YottaReal AI chatbot</li>
                    </ul>
                </div>
        """
    
    return f"""
    <html>
        <head><title>Unified Document Extractor API</title></head>
        <body style="font-family: Arial; padding: 40px; background: #f5f5f5;">
            <div style="max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h1 style="color: #333;">Unified Document Extractor API</h1>
                <p style="color: #666;">Powered by Azure Document Intelligence & Azure OpenAI</p>
                
                <div style="margin-top: 30px;">
                    <h3>Available Endpoints:</h3>
                    <ul style="line-height: 2;">
                        <li><a href="/docs" style="color: #0066cc;">📚 API Documentation</a></li>
                        <li><a href="/health" style="color: #0066cc;">❤️ Health Check</a></li>
                    </ul>
                </div>
                
                {frontend_links}
                
                <div style="margin-top: 30px;">
                    <h3>Extraction Endpoints:</h3>
                    <ul style="line-height: 2;">
                        <li><code style="background: #f0f0f0; padding: 5px;">POST /extract/invoice</code> - Invoice extraction</li>
                        <li><code style="background: #f0f0f0; padding: 5px;">POST /extract/poi</code> - Proof of Income extraction</li>
                        <li><code style="background: #f0f0f0; padding: 5px;">POST /extract/id</code> - ID Document extraction</li>
                        <li><code style="background: #f0f0f0; padding: 5px;">POST /extract/insurance</code> - Insurance Document extraction</li>
                        <li><code style="background: #f0f0f0; padding: 5px;">POST /extract/bulkinvoice</code> - Bulk Invoice extraction</li>
                        <li><code style="background: #f0f0f0; padding: 5px;">POST /extract/vendor</code> - Vendor Certificate extraction</li>
                    </ul>
                </div>
                
                <div style="margin-top: 30px; padding: 20px; background: #f0f8ff; border-left: 4px solid #0066cc;">
                    <h3>Supported Documents:</h3>
                    <ul>
                        <li><strong>Invoice:</strong> Vendor invoices, bills, receipts</li>
                        <li><strong>POI:</strong> Paystubs, employer letters, W-2, 1099, bank statements</li>
                        <li><strong>ID:</strong> Driver licenses, passports, national IDs, visas</li>
                        <li><strong>Insurance:</strong> Home insurance policies, certificates</li>
                        <li><strong>Bulk Invoice:</strong> Multiple vendor invoices in a single request</li>
                        <li><strong>Vendor:</strong> ACORD Certificate of Liability Insurance</li>
                    </ul>
                </div>
                
                <div style="margin-top: 30px; padding: 20px; background: #fff8dc; border-left: 4px solid #ffa500;">
                    <h3>Authentication:</h3>
                    <p>Include your API key in the request header:</p>
                    <code style="background: #f0f0f0; padding: 10px; display: block; margin-top: 10px;">
                        X-API-Key: your_api_key_here
                    </code>
                </div>
            </div>
        </body>
    </html>
    """

# API Documentation
@app.get("/api-docs", response_class=HTMLResponse)
async def api_documentation():
    """Serve API integration documentation"""
    try:
        with open("api-documentation.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Documentation not found</h1>"
        
# ==================== INVOICE EXTRACTION ====================

@app.post("/extract/invoice")
async def extract_invoice(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key_invoice)
):
    """Extract data from invoice document"""
    start_time = datetime.now()
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        try:
            extracted_data = extract_invoice_data(content)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        
        invoices: List[Dict[str, Any]] = [extracted_data] if isinstance(extracted_data, dict) else (extracted_data or [])
        if not invoices:
            raise HTTPException(status_code=400, detail="No invoice data returned")
        
        needs_review, review_reason = requires_human_review_invoice(invoices)
        field_counts = [count_non_null_fields(i) for i in invoices]
        field_count = min(field_counts) if field_counts else 0
        total_time = (datetime.now() - start_time).total_seconds()
        status = "needs_review" if needs_review else "success"
        
        return {
            "status": status,
            "document_type": "invoice",
            "requires_human_review": needs_review,
            "review_reason": review_reason if needs_review else None,
            "data": invoices,
            "field_count": field_count,
            "processing_time_seconds": round(total_time, 2),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        error_trace = traceback.format_exc()
        fname = getattr(file, "filename", None) or "unknown"
        raise HTTPException(status_code=500, detail={"error": error_msg, "file": fname, "traceback": error_trace[:500]})

# ==================== POI EXTRACTION ====================

@app.post("/extract/poi")
async def extract_poi(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key_poi)
):
    """Extract proof of income/employment data from document"""
    start_time = datetime.now()
    
    logger.info(f"Received POI file: {file.filename}")
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        content = await file.read()
        logger.info(f"File read: {len(content)} bytes")
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        try:
            logger.info("Starting OCR extraction...")
            extracted_data = extract_employment_data_ocr(content)
            extraction_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"OCR extraction completed in {extraction_time:.2f}s")
        except ValueError as ve:
            extraction_time = (datetime.now() - start_time).total_seconds()
            raise HTTPException(status_code=400, detail=str(ve))
        
        logger.info("Applying filter for employment verification applications...")
        extracted_data = filter_verification_applications(extracted_data)
        extracted_data = recompute_average_monthly_income(extracted_data)
        
        if not has_proof_of_employment(extracted_data):
            msg = "The provided document does not appear to be a proof of employment."
            raise HTTPException(status_code=400, detail=msg)
        
        needs_review, review_reason = requires_human_review_poi(extracted_data)
        logger.info(f"Needs review: {needs_review}, Reason: {review_reason}")
        
        vital_fields_status = get_vital_fields_status_poi(extracted_data)
        field_count = count_non_null_fields(extracted_data)
        total_time = (datetime.now() - start_time).total_seconds()
        status = "needs_review" if needs_review else "success"
        
        return {
            "status": status,
            "document_type": "poi",
            "requires_human_review": needs_review,
            "review_reason": review_reason if needs_review else None,
            "data": extracted_data,
            "field_count": field_count,
            "vital_fields_status": vital_fields_status,
            "processing_time_seconds": round(total_time, 2),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        error_trace = traceback.format_exc()
        logger.error(f"Error after {total_time:.2f}s: {error_msg}")
        logger.error(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail={"error": error_msg, "file": file.filename, "traceback": error_trace[:500]})

# ==================== ID EXTRACTION ====================

@app.post("/extract/id")
async def extract_id(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key_id)
):
    """Extract data from ID document"""
    start_time = datetime.now()
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        try:
            extracted_data = extract_id_data(content)
            
            if isinstance(extracted_data, dict):
                raw_ids: List[Dict[str, Any]] = [extracted_data]
            else:
                raw_ids = extracted_data
            
            # Filter out invalid IDs (missing both name and ID number)
            ids = filter_valid_ids(raw_ids)
            
            # If all IDs were filtered out, return error
            if not ids:
                msg = "The provided document does not appear to be an ID document."
                raise HTTPException(status_code=400, detail=msg)
                
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        
        needs_review, review_reason = requires_human_review_id(ids)
        
        all_missing_vitals = all(
            not (
                d.get("id_number") not in (None, "", [])
                or d.get("full_name") not in (None, "", [])
                or d.get("date_of_birth") not in (None, "", [])
            )
            for d in ids
        )
        
        if all_missing_vitals:
            msg = "The provided document does not appear to be an ID document."
            raise HTTPException(status_code=400, detail=msg)
        
        vital_fields = ["id_number", "full_name", "date_of_birth"]
        vital_fields_status = {
            field: all(d.get(field) not in (None, "", []) for d in ids)
            for field in vital_fields
        }
        
        field_counts = [count_non_null_fields_id(d) for d in ids]
        field_count = min(field_counts) if field_counts else 0
        total_time = (datetime.now() - start_time).total_seconds()
        status = "needs_review" if needs_review else "success"
        
        return {
            "status": status,
            "document_type": "id",
            "requires_human_review": needs_review,
            "review_reason": review_reason if needs_review else None,
            "data": ids,
            "field_count": field_count,
            "vital_fields_status": vital_fields_status,
            "processing_time_seconds": round(total_time, 2),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": error_msg, "file": file.filename, "traceback": error_trace[:500]})

# ==================== INSURANCE EXTRACTION ====================

@app.post("/extract/insurance")
async def extract_insurance(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key_insurance)
):
    """Extract data from insurance document"""
    start_time = datetime.now()
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        try:
            extracted_data = extract_insurance_data_ocr(content)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        
        needs_review, review_reason = requires_human_review_insurance(extracted_data)
        
        if all(
            extracted_data.get(field) in (None, "", [])
            for field in ["policy_number", "insured_name", "insurance_company"]
        ):
            msg = "The provided document does not appear to be an insurance document."
            raise HTTPException(status_code=400, detail=msg)
        
        vital_fields = ["policy_number", "insured_name", "insurance_company"]
        vital_fields_status = {
            field: extracted_data.get(field) not in (None, "", [])
            for field in vital_fields
        }
        
        field_count = count_non_null_fields(extracted_data)
        total_time = (datetime.now() - start_time).total_seconds()
        status = "needs_review" if needs_review else "success"
        
        return {
            "status": status,
            "document_type": "insurance",
            "requires_human_review": needs_review,
            "review_reason": review_reason if needs_review else None,
            "data": extracted_data,
            "field_count": field_count,
            "vital_fields_status": vital_fields_status,
            "processing_time_seconds": round(total_time, 2),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": error_msg, "file": file.filename, "traceback": error_trace[:500]})

# ==================== BULK INVOICE EXTRACTION ====================

@app.post("/extract/bulkinvoice")
async def extract_bulk_invoice(
    files: List[UploadFile] = File(...),
    authenticated: bool = Depends(verify_api_key_bulkinvoice)
):
    """Extract data from multiple invoice documents"""
    start_time = datetime.now()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Validate all files before processing any
    file_data: List[tuple] = []
    for file in files:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="One or more files are missing a filename."
            )
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed: '{file.filename}'. "
                       f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=400,
                detail=f"File is empty: '{file.filename}'"
            )
        if len(content) > BULK_MAX_FILE_SIZE:
            size_mb = len(content) / 1024 / 1024
            raise HTTPException(
                status_code=400,
                detail=f"File too large: '{file.filename}' ({size_mb:.1f}MB). "
                       f"Maximum allowed size is 5MB.",
            )
        file_data.append((file.filename, content))

    logger.info(
        f"Processing {len(file_data)} file(s) in parallel",
        extra={"file_count": len(file_data), "filenames": [f for f, _ in file_data]},
    )

    loop = asyncio.get_running_loop()

    async def process_one(filename: str, content: bytes) -> Dict[str, Any]:
        async with _gpt_semaphore:
            return await loop.run_in_executor(
                None, extract_bulk_invoice_data, content, filename
            )

    try:
        raw_results: List[Dict[str, Any]] = await asyncio.gather(
            *[process_one(fname, content) for fname, content in file_data]
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(
            f"Bulk invoice extraction failed: {error_msg}",
            exc_info=True,
            extra={"filenames": [f for f, _ in file_data]},
        )
        correlation_match = re.search(r"\[([a-f0-9]{8})\]", error_msg)
        correlation_id = correlation_match.group(1) if correlation_match else None
        raise HTTPException(
            status_code=400,
            detail={
                "error": error_msg,
                "correlation_id": correlation_id,
                "hint": "Use the correlation_id to find full details in server logs.",
            }
        )

    # Assign auto-incrementing IDs across all files
    company_id = 1
    invoice_id = 1
    invoice_item_id = 1
    final_results: List[Dict[str, Any]] = []

    for raw in raw_results:
        invoices: List[Dict[str, Any]] = []

        for prop in raw.get("properties", []):
            current_invoice_id = invoice_id
            invoice_items: List[Dict[str, Any]] = []

            for item in prop.get("invoiceItems", []):
                invoice_items.append({
                    "invoiceItemId": invoice_item_id,
                    "invoiceId": current_invoice_id,
                    "description": item.get("description"),
                    "quantity": item.get("quantity"),
                    "price": item.get("price"),
                    "overhead": item.get("overhead"),
                    "freight": item.get("freight"),
                    "tax": item.get("tax"),
                    "discount": item.get("discount"),
                    "totalPrice": item.get("totalPrice"),
                })
                invoice_item_id += 1

            invoices.append({
                "invoiceId": current_invoice_id,
                "propertyName": prop.get("propertyName"),
                "vendorName": raw.get("vendorName"),
                "invoiceNumber": prop.get("invoiceNumber") or raw.get("invoiceNumber"),
                "invoiceDate": prop.get("invoiceDate") or raw.get("invoiceDate"),
                "notes": prop.get("notes"),
                "uploadInvoice": raw.get("uploadInvoice"),
                "invoiceItems": invoice_items,
            })
            invoice_id += 1

        final_results.append({
            "companyId": company_id,
            "companyName": raw.get("companyName"),
            "invoices": invoices,
        })
        company_id += 1

    total_time = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"Completed {len(file_data)} file(s) in {total_time:.2f}s",
        extra={
            "file_count": len(file_data),
            "duration_seconds": round(total_time, 2),
            "total_invoices": sum(len(r.get("invoices", [])) for r in final_results),
        },
    )

    # print(json.dumps({"status": "success", "document_type": "bulkinvoice", "data": final_results}, indent=2))

    return {
        "status": "success",
        "document_type": "bulkinvoice",
        "data": final_results,
        "processing_time_seconds": round(total_time, 2),
    }

# ==================== VENDOR EXTRACTION ====================

@app.post("/extract/vendor")
async def extract_vendor(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key_vendor)
):
    """Extract data from ACORD Certificate of Liability Insurance"""
    start_time = datetime.now()

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum size is 15MB"
            )

        try:
            extracted_data = extract_vendor_data_ocr(content)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        if all(
            extracted_data.get(field) in (None, "", [])
            for field in ["insured_name"]
        ):
            msg = "The provided document does not appear to be a vendor certificate."
            raise HTTPException(status_code=400, detail=msg)

        vital_fields = ["insured_name"]
        vital_fields_status = {
            field: extracted_data.get(field) not in (None, "", [])
            for field in vital_fields
        }

        needs_review = not all(vital_fields_status.values())
        review_reason = (
            f"Missing vital field(s): {', '.join(f for f, v in vital_fields_status.items() if not v)}"
            if needs_review else None
        )

        field_count = count_non_null_fields(extracted_data)
        total_time = (datetime.now() - start_time).total_seconds()
        status = "needs_review" if needs_review else "success"

        return {
            "status": status,
            "document_type": "vendor",
            "requires_human_review": needs_review,
            "review_reason": review_reason,
            "data": extracted_data,
            "field_count": field_count,
            "vital_fields_status": vital_fields_status,
            "processing_time_seconds": round(total_time, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": error_msg, "file": file.filename, "traceback": error_trace[:500]})

@app.post("/validate/checks")
async def validate_checks(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key_checkvalidation),
):
    start_time = datetime.now()
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        try:
            result = await validate_checks_file(file.filename, content)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        total_time = (datetime.now() - start_time).total_seconds()
        return {
            "status": "success",
            "document_type": "checkvalidation",
            "data": result.model_dump(mode="json", exclude_none=True),
            "processing_time_seconds": round(total_time, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": error_msg, "file": file.filename, "traceback": error_trace[:500]})

# ==================== FRONTEND ROUTES ====================

# Resolve the React build directory (services/frontend/build/)
_FRONTEND_BUILD = Path(__file__).resolve().parent.parent / "frontend" / "build"

# Mount React static assets at /chatbot/static (matches homepage: "/chatbot")
if _FRONTEND_BUILD.exists():
    app.mount("/chatbot/static", StaticFiles(directory=str(_FRONTEND_BUILD / "static")), name="chatbot_static")

@app.get("/chat", tags=["Frontend"], summary="Open the YottaReal chatbot UI")
async def chat_frontend():
    """Redirect to the chatbot UI served at /chatbot/."""
    if not ENABLE_FRONTEND:
        raise HTTPException(status_code=404, detail="Frontend is disabled")
    return RedirectResponse(url="/chatbot/")

@app.get("/chatbot", response_class=HTMLResponse, include_in_schema=False)
@app.get("/chatbot/{path:path}", response_class=HTMLResponse, include_in_schema=False)
async def chatbot_spa(path: str = ""):
    """Serve the React SPA index.html for all /chatbot/* routes."""
    if not ENABLE_FRONTEND:
        raise HTTPException(status_code=404, detail="Frontend is disabled")
    index = _FRONTEND_BUILD / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>Chatbot not built</h1><p>Run <code>npm run build</code> in services/frontend/</p>", status_code=404)
    return HTMLResponse(index.read_text())

@app.get("/test", response_class=HTMLResponse)
async def test_page():
    """Serve test page for all document types"""
    if not ENABLE_FRONTEND:
        raise HTTPException(status_code=404, detail="Frontend is disabled")
    
    try:
        with open("../frontend/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Test page not found</h1><p>Make sure frontend/index.html exists</p>"

# ==================== CHATBOT ENDPOINTS ====================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest, authenticated: bool = Depends(verify_chatbot_api_key)):
    """Process chat messages with session uploads and indexed document retrieval."""
    try:
        if not body.session_id:
            body.session_id = str(uuid.uuid4())

        await enforce_session_rate_limit(
            session_id=body.session_id,
            action="chat",
            rate_limit=chatbot_config.RATE_LIMIT_CHAT,
        )

        logger.info(f"Chat request - Session ID: {body.session_id}, Query: {body.message}")

        session_context = []
        redis_client = await get_redis_client()

        if body.session_id:
            session_key = f"session:{body.session_id}"
            session_data = await redis_client.get(session_key)
            session_docs = json.loads(session_data) if session_data else []

            if session_data:
                await redis_client.expire(session_key, chatbot_config.SESSION_TTL_SECONDS)

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
                            "page_number": page_info['page_number'],
                        })
                else:
                    fallback_content = (doc.get("content") or "").strip()
                    if not fallback_content:
                        continue
                    session_context.append({
                        "content": fallback_content,
                        "filename": doc["filename"],
                        "source_type": "uploaded",
                        "page_number": 1,
                    })

            logger.info(f"Uploaded documents in session: {len(session_docs)} files")
        else:
            logger.info("No uploaded documents in this session")

        casual_patterns = [
            'hi', 'hello', 'hey', 'how are you', 'thanks',
            'thank you', 'bye', 'goodbye', 'good morning', 'good evening',
            'sup', "what's up", 'wassup', 'yo', 'howdy', 'good night',
        ]
        query_lower = body.message.lower().strip()
        is_casual = False
        if query_lower in casual_patterns:
            is_casual = True
        elif len(query_lower.split()) <= 2:
            if any(p in query_lower for p in casual_patterns):
                is_casual = True
        elif any(p in query_lower for p in ['how are', 'how r u', 'how r you', 'hows it going', 'how do you do']):
            is_casual = True
        elif len(query_lower.split()) == 1 and len(query_lower) <= 6:
            is_casual = True

        logger.info(f"Query type: {'Casual chat' if is_casual else 'Document query'}")

        indexed_results = []
        if not is_casual:
            logger.info("Searching company documents")
            indexed_results = await search_service.search(body.message)
            for doc in indexed_results:
                doc["source_type"] = "company"
            logger.info(f"Found {len(indexed_results)} company documents")
        else:
            logger.info("Skipping document search (casual chat)")

        if is_casual:
            all_context = []
        elif session_context:
            all_context = session_context + indexed_results[:15]
        else:
            all_context = indexed_results[:15]

        logger.info(f"Sending to LLM ({len(all_context)} document pages)")
        if not all_context and not is_casual:
            logger.warning("No documents in context for non-casual query")

        response = await llm_service.generate_response(
            query=body.message,
            context=all_context,
            session_id=body.session_id,
            has_uploads=bool(session_context),
            is_comparison=False,
        )

        source_map = {}
        for source in response["sources"]:
            filename = source.get("filename", "Unknown")
            if filename not in source_map:
                source_map[filename] = source
        unique_sources = list(source_map.values())

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
            session_id=response["session_id"],
        )

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing chat request")


@app.post("/api/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    authenticated: bool = Depends(verify_chatbot_api_key),
):
    """Upload a document, extract text, and store results in Redis session state."""
    try:
        if not session_id:
            session_id = str(uuid.uuid4())

        await enforce_session_rate_limit(
            session_id=session_id,
            action="upload",
            rate_limit=chatbot_config.RATE_LIMIT_UPLOAD,
        )

        logger.info(f"Upload request - Session ID: {session_id}, Filename: {file.filename}, Content-Type: {file.content_type}")

        await persistence_service.ensure_session(session_id)

        redis_client = await get_redis_client()
        session_key = f"session:{session_id}"
        session_data = await redis_client.get(session_key)
        current_docs = json.loads(session_data) if session_data else []

        if len(current_docs) >= chatbot_config.MAX_UPLOADS_PER_SESSION:
            logger.warning(f"Upload limit reached: {len(current_docs)}/{chatbot_config.MAX_UPLOADS_PER_SESSION}")
            raise HTTPException(
                status_code=400,
                detail=f"Upload limit reached. Maximum {chatbot_config.MAX_UPLOADS_PER_SESSION} files per session.",
            )

        if file.content_type not in CHATBOT_ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file.content_type} not supported",
            )

        file_content = await file.read()
        logger.info(f"File size: {len(file_content)} bytes")

        if len(file_content) > chatbot_config.MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds {chatbot_config.MAX_FILE_SIZE_MB}MB limit",
            )

        if not validate_chatbot_file_content(file_content, file.content_type):
            raise HTTPException(
                status_code=400,
                detail="File content does not match its declared type",
            )

        logger.info(f"Extracting text from {file.filename}")
        extraction_result = await chatbot_doc_intelligence_service.extract_text(
            file_content,
            file.filename,
        )

        if not extraction_result['success']:
            logger.error(f"Extraction failed: {extraction_result.get('error')}")
            raise HTTPException(status_code=500, detail="Failed to process uploaded file")

        logger.info(f"Extracted {len(extraction_result['text'])} characters from {extraction_result['page_count']} pages")

        blob_info = await asyncio.to_thread(
            blob_service.upload_user_file,
            file_content,
            session_id,
            file.filename,
        )
        if not blob_info:
            raise HTTPException(status_code=500, detail="Failed to store uploaded file")

        current_docs.append({
            "filename": file.filename,
            "content": extraction_result['text'],
            "page_texts": extraction_result.get('page_texts', []),
            "page_count": extraction_result['page_count'],
        })

        await redis_client.setex(
            session_key,
            chatbot_config.SESSION_TTL_SECONDS,
            json.dumps(current_docs),
        )

        upload_id = await persistence_service.save_upload(
            session_id=session_id,
            filename=file.filename,
            content_type=file.content_type,
            extraction_result=extraction_result,
            blob_info=blob_info,
        )

        logger.info(f"Stored in Redis session: {session_id}")
        logger.info(f"Session now has {len(current_docs)}/{chatbot_config.MAX_UPLOADS_PER_SESSION} documents")

        return {
            "message": "File uploaded and ready for queries!",
            "filename": file.filename,
            "session_id": session_id,
            "upload_id": upload_id,
            "pages_extracted": extraction_result['page_count'],
            "text_length": len(extraction_result['text']),
            "immediate_access": True,
            "uploads_remaining": chatbot_config.MAX_UPLOADS_PER_SESSION - len(current_docs),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in upload_document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while uploading document")


@app.post("/api/cleanup-session")
async def cleanup_session(
    request_body: CleanupRequest,
    authenticated: bool = Depends(verify_chatbot_api_key),
):
    """Delete all uploaded documents for a session from Redis."""
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
            return {"message": "Session cleaned up successfully", "session_id": session_id, "files_deleted": files_count}

        logger.warning("Session not found")
        await redis_client.delete(conversation_key)
        try:
            await persistence_service.delete_session(session_id)
        except Exception as persistence_error:
            logger.warning("Failed to delete persisted session data: %s", persistence_error)

        return {"message": "No session found", "session_id": session_id, "files_deleted": 0}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in cleanup_session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while cleaning up session")


@app.get("/api/indexer/status")
async def get_indexer_status(authenticated: bool = Depends(verify_chatbot_api_key)):
    """Return current Azure Search indexer status and latest execution metadata."""
    try:
        status = await search_service.get_indexer_status()
        return status
    except Exception as e:
        logger.exception(f"Error getting indexer status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching indexer status")


@app.post("/api/indexer/run")
async def run_indexer(authenticated: bool = Depends(verify_chatbot_api_key)):
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


# ==================== HEALTH CHECK ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "invoice": "available",
            "poi": "available",
            "id": "available",
            "insurance": "available",
            "bulkinvoice": "available",
            "vendor": "available",
            "checkvalidation": "available",
        },
        "chatbot": "available",
        "redis": "healthy",
    }
    try:
        redis_client = await get_redis_client()
        await redis_client.ping()
    except Exception as e:
        logger.warning(f"Health check degraded due to Redis issue: {e}")
        health["status"] = "degraded"
        health["redis"] = "unhealthy"
    return health

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)