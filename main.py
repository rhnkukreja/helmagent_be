import os
import io
import json
import base64
import zipfile
import tempfile
import shutil
import uuid
import gc
import traceback
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client, Client
from openai import AsyncOpenAI
from dotenv import load_dotenv
import httpx
import requests

from llm_responses import extract_text_from_image, extract_text_from_html
from utils import store_in_supabase
from routes_whatsapp import router as whatsapp_router
from routes_razorpay import router as razorpay_router
from memory_logger import log_memory_usage_to_file

# -------------------------------------------------------
load_dotenv()
print("Script started running...")

# ---------- CONFIG ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BACKEND_URL = os.getenv("BACKEND_URL")

# ---------- CLIENTS ----------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
thread_pool = ThreadPoolExecutor(max_workers=10)

# ---------- FASTAPI ----------
app = FastAPI(title="Bill Processor API – Async + Queue", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(whatsapp_router)
app.include_router(razorpay_router)

# ---------- QUEUE & WORKERS ----------
processing_queue: asyncio.Queue = asyncio.Queue()
# NUM_WORKERS = int(os.getenv("NUM_WORKERS", "5"))
NUM_WORKERS=2
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.1"))

job_status: Dict[str, Dict[str, Any]] = {}


# -------------------------------------------------------
# ---------- SUPABASE BULK UPSERT ----------
def bulk_upsert_to_supabase(extracted_data_list: List[dict]) -> List[dict]:
    """Insert many bills in one request. Only fields that exist in the table."""
    try:
        ALLOWED_FIELDS = {
            "name", "contact_number", "items_ordered", "bill_date",
            "total_amount", "org_id"
        }

        records = []
        for data in extracted_data_list:
            rec = {}
            for field in ALLOWED_FIELDS:
                if field == "items_ordered":
                    rec[field] = json.dumps(data.get(field, []))
                elif field == "bill_date":
                    rec[field] = data.get("date") or data.get("bill_date")
                elif field == "total_amount":
                    val = data.get(field)
                    rec[field] = str(val) if val not in (None, "", 0) else None
                else:
                    rec[field] = data.get(field)
            records.append(rec)

        loop = asyncio.get_event_loop()
        resp = loop.run_in_executor(None, lambda: supabase.table("bills").insert(records).execute())
        print("✅ Bulk upsert completed in Supabase.")
        return [] if resp is None else getattr(resp, 'data', []) or []
    except Exception as e:
        print(f"[SUPABASE] bulk insert error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------
# ---------- BACKGROUND WORKER ----------
async def worker():
    """Consume sub-jobs from the queue and process them."""
    print(f"[WORKER] Started")
    while True:
        try:
            sub_job = await asyncio.wait_for(processing_queue.get(), timeout=60)
            job_id = sub_job["job_id"]
            
            job_status[job_id]["status"] = "running"
            try:
                await process_sub_job(sub_job)
                job_status[job_id]["status"] = "completed"
                print(f"[WORKER] Job {job_id} completed ✅")
            except Exception as exc:
                job_status[job_id]["status"] = "failed"
                job_status[job_id]["errors"].append(str(exc))
                print(f"[WORKER] Job {job_id} failed: {exc}")
                traceback.print_exc()
            finally:
                processing_queue.task_done()
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            print(f"[WORKER] Unexpected error: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)


async def process_one_file(inner_name: str, temp_dir: str, org_id: str):
    """Process a single file (image or HTML)."""
    safe_path = os.path.normpath(os.path.join(temp_dir, inner_name.replace("\\", "/")))

    result = {
        "source_file": inner_name,
        "org_id": org_id,
        "file_type": None,
        "error": None,
        "data": None,
    }

    if not os.path.exists(safe_path):
        result["error"] = f"File not found: {inner_name}"
        return result

    ext = os.path.splitext(inner_name)[1].lower()

    try:
        # IMAGE FILES
        if ext in {".jpg", ".jpeg", ".png"}:
            with open(safe_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()

            data = await extract_text_from_image(encoded)
            result["file_type"] = "image"
            result["data"] = data
            return result

        # HTML FILES
        if ext == ".html":
            with open(safe_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()

            data = await extract_text_from_html(html)
            result["file_type"] = "html"
            result["data"] = data
            return result

        result["error"] = f"Unsupported file type: {ext}"
        return result

    except Exception as e:
        result["error"] = str(e)
        traceback.print_exc()
        return result


async def process_sub_job(sub_job: dict):
    """Process a batch of files from a ZIP."""
    temp_dir = sub_job["temp_dir"]
    file_list = sub_job["files_list"]
    org_id = sub_job["org_id"]

    tasks = [process_one_file(inner_name, temp_dir, org_id) for inner_name in file_list]
    results = await asyncio.gather(*tasks)

    extracted_batch = []

    for result in results:
        if result["error"]:
            sub_job["errors"].append(f"{result['source_file']}: {result['error']}")
            continue

        data = result["data"]
        if not data:
            sub_job["errors"].append(f"{result['source_file']}: No data returned")
            continue

        data["org_id"] = org_id
        data["source_file"] = result["source_file"]
        data["file_type"] = result["file_type"]

        extracted_batch.append(data)
        sub_job["processed_files"] += 1

    if extracted_batch:
        try:
            inserted = bulk_upsert_to_supabase(extracted_batch)
            sub_job["results"].extend(inserted)
        except Exception as e:
            sub_job["errors"].append(f"DB save failed: {e}")


# -------------------------------------------------------
# ---------- ENDPOINTS ----------
@app.post("/process-bill/")
@log_memory_usage_to_file
async def process_bill(
    files: List[UploadFile] = File(...),
    org_id: str = Form(...),
):
    """Process bills from images, HTMLs, or ZIPs."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file required.")
    if not org_id:
        raise HTTPException(status_code=400, detail="org_id required.")

    results = []

    for file in files:
        filename = file.filename.lower()
        content_type = file.content_type or ""

        # ---------- SINGLE IMAGE ----------
        if content_type.startswith("image/") or filename.endswith((".jpg", ".jpeg", ".png")):
            try:
                image_bytes = await file.read()
                encoded = base64.b64encode(image_bytes).decode()
                data = await extract_text_from_image(encoded)
                data["file_type"] = "image"
                data["org_id"] = org_id
                stored = store_in_supabase(data)
                results.append({"filename": file.filename, "status": "success", "data": stored})
            except Exception as e:
                results.append({"filename": file.filename, "status": "error", "error": str(e)})
            continue

        # ---------- SINGLE HTML ----------
        if filename.endswith(".html"):
            try:
                html_bytes = await file.read()
                html = html_bytes.decode("utf-8", errors="ignore")
                data = await extract_text_from_html(html)
                data["file_type"] = "html"
                data["org_id"] = org_id
                stored = store_in_supabase(data)
                results.append({"filename": file.filename, "status": "success", "data": stored})
            except Exception as e:
                results.append({"filename": file.filename, "status": "error", "error": str(e)})
            continue

        # ---------- ZIP ----------
        if filename.endswith(".zip"):
            try:
                job_id = str(uuid.uuid4())
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "upload.zip")

                with open(zip_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)

                supported_infos = []
                with zipfile.ZipFile(zip_path, "r") as z:
                    for info in z.infolist():
                        if info.is_dir():
                            continue
                        name = info.filename.replace("\\", "/").lstrip("./")
                        if name.lower().endswith((".jpg", ".jpeg", ".png", ".html")) and not name.startswith("__MACOSX"):
                            supported_infos.append((info, name))

                total_files = len(supported_infos)
                if total_files == 0:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    results.append({"filename": filename, "status": "error", "error": "No supported files in ZIP"})
                    continue

                CHUNK_SIZE = 100
                successfully_processed = 0
                all_errors = []

                print(f"[ZIP] Processing {total_files} files in chunks of {CHUNK_SIZE}...")

                with zipfile.ZipFile(zip_path, "r") as z:
                    for chunk_idx, i in enumerate(range(0, total_files, CHUNK_SIZE)):
                        chunk = supported_infos[i:i + CHUNK_SIZE]
                        print(f"[ZIP] Chunk {chunk_idx + 1} — {len(chunk)} files")

                        extracted_paths = []
                        for zip_info, name in chunk:
                            try:
                                path = z.extract(zip_info, temp_dir)
                                extracted_paths.append((name, path))
                            except Exception as e:
                                all_errors.append(f"{name}: Extract failed — {e}")

                        tasks = [process_one_file(name, temp_dir, org_id) for name, _ in extracted_paths]
                        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

                        batch_to_save = []
                        for (name, path), result in zip(extracted_paths, chunk_results):
                            if isinstance(result, Exception):
                                all_errors.append(f"{name}: {str(result)}")
                                continue
                            if result["error"]:
                                all_errors.append(f"{name}: {result['error']}")
                                continue

                            if result["data"]:
                                data = result["data"]
                                data["org_id"] = org_id
                                data["source_file"] = name
                                data["file_type"] = result["file_type"]
                                batch_to_save.append(data)
                                successfully_processed += 1

                            try:
                                os.remove(path)
                            except:
                                pass

                        if batch_to_save:
                            try:
                                bulk_upsert_to_supabase(batch_to_save)
                            except Exception as e:
                                all_errors.append(f"DB save failed: {e}")

                        del chunk_results, batch_to_save, extracted_paths
                        gc.collect()

                shutil.rmtree(temp_dir, ignore_errors=True)

                results.append({
                    "filename": filename,
                    "parent_job_id": job_id,
                    "total_files": total_files,
                    "successfully_processed": successfully_processed,
                    "errors": all_errors or None,
                    "status": "completed",
                })
            except Exception as e:
                print(f"[ZIP] Error: {e}")
                traceback.print_exc()
                results.append({"filename": filename, "status": "error", "error": str(e)})
            continue

        results.append({"filename": filename, "status": "error", "error": "Unsupported file type"})

    return JSONResponse(content={"status": "success", "files_processed": len(results), "details": results})


@app.get("/queue-status/{job_id}")
@log_memory_usage_to_file
async def get_queue_status(job_id: str):
    """Poll job status."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    info = job_status[job_id].copy()
    for k in ["zip_path", "temp_dir", "files_list"]:
        info.pop(k, None)

    return info


@app.get("/dashboard/{org_id}")
@log_memory_usage_to_file
async def get_dashboard_data(org_id: str):
    """Fetch dashboard data from Supabase."""
    try:
        loop = asyncio.get_event_loop()
        org_response = await loop.run_in_executor(
            None,
            lambda: supabase.table("organizations").select("*").eq("org_id", str(org_id)).execute()
        )

        org = org_response.data[0] if org_response.data else None

        if not org:
            org = {
                "name": "Unnamed Restaurant",
                "owner_name": "Owner Not Set",
                "phone": "N/A",
                "total_reviews": 0,
                "active_conversations": 0,
                "open_issues": 0,
                "avg_rating": 0.0,
            }

        activities_response = await loop.run_in_executor(
            None,
            lambda: supabase.table("activities").select("*").eq("org_id", org_id).order("created_at", desc=True).limit(10).execute()
        )
        activities = activities_response.data or []

        dashboard_data = {
            "organization": {
                "name": org.get("name"),
                "owner_name": org.get("owner_name"),
                "phone": org.get("phone"),
                "avg_rating": float(org.get("avg_rating", 0.0)) if org.get("avg_rating") else 0.0,
            },
            "todayStats": {
                "reviews": {"count": org.get("total_reviews", 0), "change": "+0 today", "positive": True},
                "conversations": {"count": org.get("active_conversations", 0), "activeNow": 0},
                "issues": {"count": org.get("open_issues", 0), "label": "open issues"},
                "rating": {"value": float(org.get("avg_rating", 0.0)) if org.get("avg_rating") else 0.0, "change": "No change"},
            },
            "activities": activities,
        }

        return {"success": True, "data": dashboard_data}

    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bills/{org_id}")
@log_memory_usage_to_file
async def get_bills(org_id: str):
    """Fetch bills from Supabase."""
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("bills").select("*").eq("org_id", org_id).order("id", desc=True).execute()
        )

        bills = response.data or []

        for bill in bills:
            bill_date = bill.get("bill_date")
            if bill_date:
                try:
                    formatted = datetime.strptime(bill_date, "%Y-%m-%d").strftime("%b %d, %Y")
                    bill["order_date"] = formatted
                except Exception:
                    bill["order_date"] = bill_date
            else:
                bill["order_date"] = "—"

            bill["total_amount"] = str(bill.get("total_amount") or "")

        return {"success": True, "data": bills}

    except Exception as e:
        print(f"❌ Error fetching bills: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/whatsapp/status/{org_id}")
@log_memory_usage_to_file
async def get_whatsapp_status(org_id: str):
    """Fetch WhatsApp connection status."""
    try:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            lambda: supabase.table("whatsapp_connections").select("*").eq("org_id", org_id).execute()
        )
        if not res.data:
            return {"status": "not_connected"}
        return res.data[0]
    except Exception as e:
        print(f"❌ Error fetching WhatsApp status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel

SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


def get_restaurant_name(uuid: str) -> str:
    """Fetch restaurant name from Supabase Auth."""
    try:
        url = f"{SUPABASE_URL}/auth/v1/admin/users/{uuid}"
        headers = {"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        res = requests.get(url, headers=headers, timeout=5)
        res.raise_for_status()
        user = res.json()
        metadata = user.get("user_metadata", {})
        display_name = (
            metadata.get("display_name")
            or metadata.get("restaurant_name")
            or metadata.get("full_name")
            or user.get("email")
        )
        return display_name.strip() if display_name else "Unknown"
    except Exception as e:
        print(f"❌ Error fetching restaurant name: {e}")
        return "Unknown"


class GenerateMessageRequest(BaseModel):
    org_id: str
    name: str
    phone: str
    items_ordered: str
    order_date: str
    total_amount: str


@app.post("/generate-message")
@log_memory_usage_to_file
async def generate_message(request: GenerateMessageRequest):
    """Generate personalized WhatsApp message using AI."""
    try:
        print("\n" + "="*60)
        print("🤖 GENERATING AI MESSAGE")
        print("="*60)
        rest_name = get_restaurant_name(request.org_id)

        prompt = f"""
You are a restaurant manager writing a personalized WhatsApp message to a customer after their visit.
Restaurant: {rest_name}
Customer: {request.name}
Order Date: {request.order_date}
Items: {request.items_ordered}
Amount: ₹{request.total_amount}

Generate a warm, friendly message asking for feedback (under 150 words). Include:
1. Greeting addressing customer appropriately
2. Thank them for visiting
3. Mention 1-2 dishes they ordered
4. Polite feedback request
5. End with: "Warm regards,\\n[Restaurant Name] Team ❤️"

Output ONLY the message text.
"""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly restaurant manager."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        ai_message = response.choices[0].message.content.strip()
        print(f"✅ Message generated\n{ai_message}\n" + "="*60)

        return {"success": True, "message": ai_message, "customer": {"name": request.name, "phone": request.phone}}

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@app.get("/conversations/{org_id}")
@log_memory_usage_to_file
async def get_conversations(org_id: str):
    """Fetch conversations."""
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("conversations").select("*").eq("session_id", org_id).order("created_at", desc=True).execute()
        )

        if not response.data:
            return {"success": True, "data": []}

        conversations = []
        for row in response.data:
            message_list = row.get("message_list", [])
            if not isinstance(message_list, str):
                try:
                    message_list = json.dumps(message_list)
                except Exception:
                    message_list = "[]"

            conversations.append({
                "id": row.get("id"),
                "name": row.get("name"),
                "phone": row.get("phone"),
                "order_date": row.get("order_date"),
                "message_list": message_list,
                "created_at": row.get("created_at"),
                "bill_price": row.get("bill_price"),
                "session_id": row.get("session_id"),
            })

        return {"success": True, "data": conversations}

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class GoogleReviewLinkRequest(BaseModel):
    org_id: str
    link: str


@app.post("/settings/save-google-review")
@log_memory_usage_to_file
async def save_google_review_link(request: GoogleReviewLinkRequest):
    """Save Google Review link."""
    try:
        org_id = str(request.org_id).strip()
        review_link = request.link.strip()

        if not org_id or not review_link:
            raise HTTPException(status_code=400, detail="Missing org_id or link")

        loop = asyncio.get_event_loop()
        existing = await loop.run_in_executor(
            None,
            lambda: supabase.table("organizations").select("*").eq("org_id", org_id).execute()
        )

        if not existing.data:
            raise HTTPException(status_code=404, detail=f"Organization not found")

        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("organizations").update({"google_review_link": review_link}).eq("org_id", org_id).execute()
        )

        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to save link")

        return {"message": "Google Review link saved successfully!", "data": response.data[0]}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/settings/google-review/{org_id}")
@log_memory_usage_to_file
async def get_google_review_link(org_id: str):
    """Fetch Google Review link."""
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: supabase.table("organizations").select("google_review_link").eq("org_id", org_id).execute()
        )

        if not response.data:
            return {"message": "Organization not found", "link": None}

        review_link = response.data[0].get("google_review_link")
        if not review_link:
            return {"message": "No review link found", "link": None}

        return {"link": review_link}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"message": "Internal server error", "error": str(e)}


# ---------- START WORKERS ----------
@app.on_event("startup")
async def startup_event():
    print(f"🚀 Starting {NUM_WORKERS} background workers...")
    for i in range(NUM_WORKERS):
        task = asyncio.create_task(worker())
        print(f"✅ Worker {i+1} created")


@app.on_event("shutdown")
async def shutdown_event():
    print("⏹️  Shutting down...")
    await processing_queue.join()


# ---------- ROOT ----------
@app.get("/")
async def root():
    return {"message": "RepAgent Bill Processor API Running ✅"}


@app.head("/")
def head_root():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)