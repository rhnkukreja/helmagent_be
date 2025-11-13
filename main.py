import os
import io
import json
import base64
import zipfile
import tempfile
import shutil
import uuid
import traceback
from typing import List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client, Client
from openai import AsyncOpenAI
from dotenv import load_dotenv
import httpx
import requests
import asyncio

# ---- custom modules (keep your existing implementations) ----
from llm_responses import extract_text_from_image, extract_text_from_html
from utils import store_in_supabase  # kept for fallback / single-record paths
from routes_whatsapp import router as whatsapp_router
from memory_logger import log_memory_usage

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

# ---------- QUEUE & WORKERS ----------
processing_queue: asyncio.Queue = asyncio.Queue()
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "5"))      # tune per OpenAI tier
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))        # files per GPT call
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.1"))

# In-memory job tracker (replace with Redis for production)
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

        resp = supabase.table("bills").insert(records).execute()
        print("Bulk upsert response completed in Supabase.")
        return resp.data or []
    except Exception as e:
        print(f"[SUPABASE] bulk insert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------
# ---------- BACKGROUND WORKER ----------
async def worker():
    """Consume sub-jobs from the queue and process them."""
    while True:
        sub_job = await processing_queue.get()
        job_id = sub_job["job_id"]
        completion_event = sub_job.get("completion_event")
        
        job_status[job_id]["status"] = "running"
        try:
            await process_sub_job(sub_job)
            job_status[job_id]["status"] = "completed"
        except Exception as exc:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["errors"].append(str(exc))
            print(f"[WORKER] sub-job {job_id} failed: {exc}")
        finally:
            processing_queue.task_done()
            # Signal completion
            if completion_event:
                completion_event.set()

async def process_one_file(inner_name: str, temp_dir: str, org_id: str):
    safe_path = os.path.normpath(os.path.join(temp_dir, inner_name.replace("\\", "/")))

    result = {
        "source_file": inner_name,
        "org_id": org_id,
        "file_type": None,
        "error": None,
        "data": None,
    }

    # File missing
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

        # Unsupported file type
        result["error"] = f"Unsupported file type: {ext}"
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


async def process_sub_job(sub_job: dict):
    temp_dir = sub_job["temp_dir"]
    file_list = sub_job["files_list"]
    org_id = sub_job["org_id"]

    # Create parallel tasks for ALL files
    tasks = [
        process_one_file(inner_name, temp_dir, org_id)
        for inner_name in file_list
    ]

    # Run all tasks concurrently
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

    # After all GPT calls completed → save results
    if extracted_batch:
        inserted = bulk_upsert_to_supabase(extracted_batch)
        sub_job["results"].extend(inserted)

    # ---- cleanup this sub-job's temp files (parent cleans the whole dir) ----
    # (optional per-sub-job cleanup can be added here)


# -------------------------------------------------------
# ---------- ENDPOINTS ----------
@app.post("/process-bill/")
@log_memory_usage
async def process_bill(
    files: List[UploadFile] = File(...),
    org_id: str = Form(...),
):
    """
    Accept **any number** of images, HTMLs or ZIPs.
    - Single files → immediate extraction.
    - ZIP → queue sub-jobs but wait for all to complete before returning.
    Returns results after all processing is done.
    """
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
            image_bytes = await file.read()
            encoded = base64.b64encode(image_bytes).decode()
            data = await extract_text_from_image(encoded)
            data["file_type"] = "image"
            data["org_id"] = org_id
            stored = store_in_supabase(data)
            results.append({"filename": file.filename, "status": "success", "data": stored})
            continue

        # ---------- SINGLE HTML ----------
        if filename.endswith(".html"):
            html_bytes = await file.read()
            html = html_bytes.decode("utf-8", errors="ignore")
            data = await extract_text_from_html(html)
            data["file_type"] = "html"
            data["org_id"] = org_id
            stored = store_in_supabase(data)
            results.append({"filename": file.filename, "status": "success", "data": stored})
            continue

        # ---------- ZIP ----------
        if filename.endswith(".zip"):
            job_id = str(uuid.uuid4())
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, filename)
            with open(zip_path, "wb") as f:
                f.write(await file.read())

            # ---- extract supported files safely ----
            supported: List[str] = []
            with zipfile.ZipFile(zip_path, "r") as z:
                for member in z.infolist():
                    if member.is_dir():
                        continue
                    name = member.filename.replace("\\", "/").lstrip("./")
                    if name.lower().endswith((".jpg", ".jpeg", ".png", ".html")):
                        z.extract(member, temp_dir)
                        safe_path = os.path.normpath(os.path.join(temp_dir, name))
                        if os.path.exists(safe_path):
                            supported.append(name)

            if not supported:
                shutil.rmtree(temp_dir, ignore_errors=True)
                results.append({"filename": filename, "status": "error", "error": "No supported files in ZIP"})
                continue

            # ---- split into sub-jobs ----
            sub_jobs = []
            completion_events = {}  # Track completion of each sub-job
            
            for i in range(0, len(supported), BATCH_SIZE):
                batch = supported[i:i + BATCH_SIZE]
                sub_id = f"{job_id}-{i // BATCH_SIZE + 1}"
                
                # Create an event to signal completion
                completion_event = asyncio.Event()
                completion_events[sub_id] = completion_event
                
                sub_job = {
                    "job_id": sub_id,
                    "parent_zip_id": job_id,
                    "zip_path": zip_path,
                    "temp_dir": temp_dir,
                    "org_id": org_id,
                    "files_list": batch,
                    "total_files": len(batch),
                    "processed_files": 0,
                    "results": [],
                    "status": "queued",
                    "errors": [],
                    "completion_event": completion_event,  # Add event to sub-job
                }
                job_status[sub_id] = sub_job
                await processing_queue.put(sub_job)
                sub_jobs.append(sub_id)

            # parent job tracks all sub-jobs
            job_status[job_id] = {
                "job_id": job_id,
                "type": "zip",
                "temp_dir": temp_dir,
                "sub_jobs": sub_jobs,
                "status": "processing",
                "total_files": len(supported),
                "processed_files": 0,
                "results": [],
                "errors": [],
            }

            # ---- WAIT for all sub-jobs to complete ----
            await asyncio.gather(*[event.wait() for event in completion_events.values()])

            # ---- Collect all results ----
            all_results = []
            all_errors = []
            for sub_id in sub_jobs:
                sub_job_data = job_status[sub_id]
                all_results.extend(sub_job_data["results"])
                all_errors.extend(sub_job_data["errors"])
                job_status[job_id]["processed_files"] += sub_job_data["processed_files"]

            job_status[job_id]["status"] = "completed"
            job_status[job_id]["results"] = all_results

            results.append({
                "filename": filename,
                "parent_job_id": job_id,
                "total_files": len(supported),
                "processed_files": job_status[job_id]["processed_files"],
                "inserted_records": len(all_results),
                "status": "completed",
                "data": all_results,
                "errors": all_errors if all_errors else None
            })
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            continue

        # ---------- UNSUPPORTED ----------
        results.append({"filename": filename, "status": "error", "error": "Unsupported file type"})

    return JSONResponse(content={"status": "success", "files_processed": len(results), "details": results})

@app.get("/queue-status/{job_id}")
@log_memory_usage
async def get_queue_status(job_id: str):
    """Poll any job (parent ZIP or sub-job)."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    info = job_status[job_id].copy()
    # hide heavy fields from response
    for k in ["zip_path", "temp_dir", "files_list"]:
        info.pop(k, None)

    # if it's a parent ZIP, aggregate sub-job stats
    if info.get("type") == "zip":
        sub_stats = []
        processed = 0
        total = 0
        for sub_id in info.get("sub_jobs", []):
            sub = job_status.get(sub_id, {})
            sub_stats.append({
                "sub_job_id": sub_id,
                "status": sub.get("status", "unknown"),
                "processed": sub.get("processed_files", 0),
                "total": sub.get("total_files", 0),
                "errors": sub.get("errors", []),
            })
            processed += sub.get("processed_files", 0)
            total += sub.get("total_files", 0)
        info["sub_job_details"] = sub_stats
        info["processed_files"] = processed
        info["total_files"] = total

    return info


@app.get("/dashboard/{org_id}")
@log_memory_usage
async def get_dashboard_data(org_id: str):
    try:
        print("Fetching organization for org_id:", org_id)
        print("Supabase org_id type:", type(org_id), "value:", org_id)

        # --- fetch safely ---
        query = (
            supabase.table("organizations")
            .select("*")
            .eq("org_id", str(org_id))  # Supabase client should handle UUID conversion
        )

        # Use `.execute()` and check if data is empty
        try:
            org_response = query.execute()
            print("Query response:", org_response.data)  # Debug line
            # Check if we actually got data
            org = org_response.data[0] if org_response.data else None
        except Exception as inner_e:
            print("Query failed:", inner_e)
            org = None

        if not org:
            print("⚠️ No organization found in Supabase for org_id:", org_id)
            # return dummy default if nothing found
            org = {
                "name": "Unnamed Restaurant",
                "owner_name": "Owner Not Set",
                "phone": "N/A",
                "total_reviews": 0,
                "active_conversations": 0,
                "open_issues": 0,
                "avg_rating": 0.0,
            }

        # --- Fetch activities (safe) ---
        activities_response = (
            supabase.table("activities")
            .select("*")
            .eq("org_id", org_id)
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        activities = getattr(activities_response, "data", []) or []
        print("Fetched activities:", activities)

        dashboard_data = {
            "organization": {
                "name": org.get("name"),
                "owner_name": org.get("owner_name"),
                "phone": org.get("phone"),
                "avg_rating": float(org.get("avg_rating", 0.0)) if org.get("avg_rating") else 0.0,
            },
            "todayStats": {
                "reviews": {
                    "count": org.get("total_reviews", 0),
                    "change": "+0 today",
                    "positive": True,
                },
                "conversations": {
                    "count": org.get("active_conversations", 0),
                    "activeNow": 0,
                },
                "issues": {
                    "count": org.get("open_issues", 0),
                    "label": "open issues",
                },
                "rating": {
                    "value": float(org.get("avg_rating", 0.0)) if org.get("avg_rating") else 0.0,
                    "change": "No change",
                },
            },
            "activities": activities,
        }

        return {"success": True, "data": dashboard_data}

    except Exception as e:
        import traceback
        print("Error in dashboard:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/bills/{org_id}")
@log_memory_usage
async def get_bills(org_id: str):
    """
    Fetch customer bills from Supabase for a specific organization.
    Includes formatted order date and all bill details.
    """
    try:
        print(f"Fetching bills for org_id: {org_id}")
        response = (
            supabase.table("bills")
            .select("*")
            .eq("org_id", org_id)
            .order("id", desc=True)
            .execute()
        )

        bills = response.data or []
        print(f"Fetched {len(bills)} bills for org_id={org_id}")

        # 🧩 Add readable date and total formatting
        for bill in bills:
            # Handle bill_date (format for frontend)
            bill_date = bill.get("bill_date")
            if bill_date:
                try:
                    # Convert to "Oct 28, 2025" format
                    formatted = datetime.strptime(bill_date, "%Y-%m-%d").strftime("%b %d, %Y")
                    bill["order_date"] = formatted
                except Exception:
                    bill["order_date"] = bill_date  # fallback to raw
            else:
                bill["order_date"] = "—"

            # Ensure total_amount is numeric (avoid None)
            bill["total_amount"] = str(bill.get("total_amount") or "")

        return {"success": True, "data": bills}

    except Exception as e:
        print("Error fetching bills:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import Request
import traceback
import json

@app.get("/whatsapp/status/{org_id}")
@log_memory_usage
async def get_whatsapp_status(org_id: str):
    res = supabase.table("whatsapp_connections").select("*").eq("org_id", org_id).execute()
    
    if not res.data:
        return {"status": "not_connected"}
    return res.data[0]



from pydantic import BaseModel
from datetime import datetime
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
# Initialize OpenAI client

def get_restaurant_name(uuid: str) -> str:
    """
    Fetches the user's display name from Supabase Auth using UUID.
    Reads from user_metadata['display_name'] (or falls back to full_name / restaurant_name / email).
    """
    try:
        url = f"{SUPABASE_URL}/auth/v1/admin/users/{uuid}"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }

        res = requests.get(url, headers=headers)
        res.raise_for_status()
        user = res.json()

        # Extract from nested user_metadata
        metadata = user.get("user_metadata", {})
        display_name = (
            metadata.get("display_name")
            or metadata.get("restaurant_name")
            or metadata.get("full_name")
            or user.get("email")
        )

        return display_name.strip() if display_name else "Unknown"
    
    except Exception as e:
        print(f"❌ Error fetching display_name for {uuid}: {e}")
        return "Unknown"

class GenerateMessageRequest(BaseModel):
    org_id: str  
    name: str
    phone: str
    items_ordered: str
    order_date: str
    total_amount: float


@app.post("/generate-message")
@log_memory_usage
async def generate_message(request: GenerateMessageRequest):
    """
    Generate personalized WhatsApp message using AI based on customer order data
    """
    try:
        print("\n" + "="*60)
        print("🤖 GENERATING AI MESSAGE")
        print("="*60)
        print(f"Organization ID: {request.org_id}")
        print(f"Customer: {request.name}")
        print(f"Phone: {request.phone}")
        print(f"Items: {request.items_ordered}")
        print(f"Date: {request.order_date}")
        print(f"Amount: ₹{request.total_amount}")
        rest_name = get_restaurant_name(request.org_id)

        prompt = f"""
            You are a restaurant manager writing a personalized WhatsApp message to a customer after their visit.
            Restaurant Details:
            - Name: {rest_name}
            Customer Details:
            - Name: {request.name}
            - Order Date: {request.order_date}
            - Items Ordered: {request.items_ordered}
            - Total Amount: ₹{request.total_amount}

            Goal:
            Generate a warm, professional, and friendly WhatsApp message asking for customer feedback.

            Guidelines:
            1. Address the customer as:
            - "{request.name} Sir" if the name sounds male
            - "{request.name} Ma’am" if the name sounds female
            - "Dear Guest" if gender is uncertain
            (Always include greeting like “Hello” or “Dear”)
            2. Use a conversational yet polished tone with 1–2 emojis — not overly casual, not overly formal.
            3. Thank them sincerely for visiting and mention their order date naturally.
            4. Highlight 1–2 dishes they ordered and include a short chef-inspired detail 
            (e.g., “Our Butter Chicken is made with hand-ground spices for that authentic flavor.”)
            5. Politely ask how their experience was and encourage them to share feedback.
            6. Keep the message concise (under 150 words).
            7. End with this exact closing format:

            Warm regards,  
            [Restaurant Name] Team ❤️

            8. Never mention ‘Beef’.
            9. Output only the WhatsApp message text — no explanations, no labels.

            """


        # Call OpenAI API
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-4o for better quality
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly restaurant manager writing personalized WhatsApp messages to customers. Be warm, genuine, and encourage honest feedback."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=300
        )

        ai_message = response.choices[0].message.content.strip()
        
        print(f"\n✅ AI Message Generated:")
        print(ai_message)
        print("="*60 + "\n")
        
        return {
            "success": True,
            "message": ai_message,
            "customer": {
                "name": request.name,
                "phone": request.phone
            }
        }
        
    except Exception as e:
        print(f"\n❌ Error generating message: {str(e)}")
        return JSONResponse(
            content={
                "success": False,
                "error": f"Failed to generate message: {str(e)}"
            },
            status_code=500
        )

async def generate_response(chat_history: List[Dict[str, Any]]) -> str:
    """
    Send chat history to backend LLM to generate a new response.
    Assumes backend endpoint /generate-response expects {"messages": [{"role": str, "content": str}, ...]}
    with oldest message first.
    """
    # Prepare messages for LLM: reverse to chronological order (oldest first), map to roles
    messages_reversed = chat_history[::-1]  # Reverse: oldest first
    llm_messages = []
    for msg in messages_reversed:
        role = "assistant" if msg.get("is_sender", False) else "user"
        content = msg.get("text", "")
        if content:  # Skip empty
            llm_messages.append({"role": role, "content": content})

    print(f"\n🤖 Generating response based on {len(llm_messages)} messages...")
    print(f"Last few for context: {json.dumps(llm_messages[-3:], indent=2)}")  # Log last 3

    payload = {"messages": llm_messages}

    try:
            generated_msg="Sample generated response."
            return generated_msg
        # else:
        #     print(f"❌ LLM generation error: {response.status_code} - {response.text}")
        #     raise Exception(f"Failed to generate response: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception generating response: {str(e)}")
        # Fallback message
        return "Thanks for your message! How can I assist you today?"

from fastapi import FastAPI, Request
from typing import Any, Dict

@app.get("/conversations/{org_id}")
@log_memory_usage
async def get_conversations(org_id: str):
    """
    Step 1: Fetch account_id from whatsapp_connections using org_id.
    Step 2: Fetch conversations using account_id.
    Step 3: Clean data and return JSON-stringified message_list.
    """
    try:
        # Step 1 — get account_id
        """ 
        connection = (
            supabase.table("whatsapp_connections")
            .select("account_id")
            .eq("org_id", org_id)
            .single()
            .execute()
        )

        if not connection.data or not connection.data.get("account_id"):
            raise HTTPException(status_code=404, detail="No WhatsApp connection found for this organization")

        account_id = connection.data["account_id"]
        print("account_id:", account_id)
        """
        session_id = org_id
        # Step 2 — fetch conversations
        response = (
            supabase.table("conversations")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .execute()
        )

        if not response.data:
            return {"success": True, "data": []}

        # Step 3 — clean data
        conversations = []
        for row in response.data:
            # Ensure message_list is serialized properly
            message_list = row.get("message_list", [])
            if not isinstance(message_list, str):
                try:
                    message_list = json.dumps(message_list)
                except Exception:
                    message_list = "[]"

            conversations.append({
                "id": row.get("id"),
                "name": row.get("name") or None,
                "phone": row.get("phone"),
                "order_date": row.get("order_date"),
                "message_list": message_list,
                "created_at": row.get("created_at"),
                "bill_price": row.get("bill_price"),
                "session_id": row.get("session_id"),
            })

        print("✅ Conversations prepared:", len(conversations))
        print(conversations)
        return {"success": True, "data": conversations}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")

"""
@app.post("/disconnect")
async def disconnect_whatsapp(request: Request):
    
    Disconnect a WhatsApp account:
    - Fetches the account_id from Supabase.
    - Calls Unipile DELETE API to remove the connection.
    - Clears account_id and status in Supabase.
    
    try:
        body = await request.json()
        org_id = body.get("org_id")

        if not org_id:
            raise HTTPException(status_code=400, detail="Missing org_id")

        print(f"🔹 Disconnecting WhatsApp for org_id: {org_id}")

        # 1️⃣ Fetch the account_id from Supabase
        res = supabase.table("whatsapp_connections").select("account_id").eq("org_id", org_id).execute()
        if not res.data or not res.data[0].get("account_id"):
            raise HTTPException(status_code=404, detail="No WhatsApp account linked for this organization")

        account_id = res.data[0]["account_id"]
        print(f"✅ Found account_id: {account_id}")

        # 2️⃣ Call Unipile DELETE API
        delete_url = f"{UNIPILE_BASE_URL}/api/v1/accounts/{account_id}"
        headers = {
            "accept": "application/json",
            "X-API-KEY": UNIPILE_API_KEY,
        }

        print(f"🔸 Sending DELETE to {delete_url}")
        response = requests.delete(delete_url, headers=headers)

        if response.status_code not in [200, 204]:
            print(f"❌ Failed to delete account from Unipile: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Unipile delete failed: {response.text}"
            )

        print(f"🗑️ Unipile account deleted successfully.")

        # 3️⃣ Update Supabase — clear account_id & status
        update_res = supabase.table("whatsapp_connections").upsert(
            {
                "org_id": org_id,
                "account_id": "",
                "status": "",
                "last_updated_at": datetime.utcnow().isoformat()
            },
            on_conflict="org_id"
        ).execute()

        print(f"🧹 Supabase updated successfully: {update_res.data}")

        return {
            "success": True,
            "message": "WhatsApp account disconnected successfully",
            "org_id": org_id
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Error disconnecting WhatsApp: {e}")
        return {
            "success": False,
            "error": str(e)
        }
"""
class GoogleReviewLinkRequest(BaseModel):
    org_id: str
    link: str


@app.post("/settings/save-google-review")
@log_memory_usage
async def save_google_review_link(request: GoogleReviewLinkRequest):
    """
    Save or update the Google Review link for the organization
    """
    try:
        org_id = str(request.org_id).strip()
        review_link = request.link.strip()
        
        if not org_id or not review_link:
            raise HTTPException(status_code=400, detail="Missing org_id or link")
        
        # ✅ Check if organization exists
        existing = supabase.table("organizations")\
            .select("*")\
            .eq("org_id", org_id)\
            .execute()
        
        print(f"Query result: {existing.data}")
        
        if not existing.data or len(existing.data) == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"Organization not found with org_id: {org_id}"
            )
        
        # ✅ Update the review link (NOT upsert, since org already exists)
        response = supabase.table("organizations")\
            .update({"google_review_link": review_link})\
            .eq("org_id", org_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(
                status_code=500, 
                detail="Failed to save Google Review link"
            )
        
        return {
            "message": "Google Review link saved successfully!", 
            "data": response.data[0]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error saving review link: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/settings/google-review/{org_id}")
@log_memory_usage
async def get_google_review_link(org_id: str):
    """
    Fetch the saved Google Review link for an organization.
    Returns a message if the link is None or empty instead of 404 error.
    """
    try:
        response = supabase.table("organizations") \
            .select("google_review_link") \
            .eq("org_id", org_id) \
            .execute()
        
        if not response.data:
            return {"message": "Organization not found", "link": None}
        
        review_link = response.data[0].get("google_review_link")
        
        if not review_link:
            return {"message": "No review link found", "link": None}
        
        return {"link": review_link}
    
    except Exception as e:
        print("❌ Error fetching review link:", e)
        return {"message": "Internal server error", "error": str(e)}


# ---------- START WORKERS ----------
@app.on_event("startup")
async def startup_event():
    print(f"Starting {NUM_WORKERS} background workers...")
    for _ in range(NUM_WORKERS):
        asyncio.create_task(worker())

    # optional: clean old temp dirs on startup
    # (omitted for brevity)


@app.on_event("shutdown")
async def shutdown_event():
    await processing_queue.join()
    # clean any remaining temp dirs
    for job in list(job_status.values()):
        if "temp_dir" in job:
            shutil.rmtree(job["temp_dir"], ignore_errors=True)


# -------------------------------------------------------
# ---------- ROOT ----------
@app.get("/")
async def root():
    return {"message": "Enhanced Async Bill OCR API Running"}

@app.head("/")
def head_root():
    return {"status": "ok"}


# -------------------------------------------------------
# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
