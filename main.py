# --- bill OCR dependencies ---
# Add at top 3rd algorithm file main_new.py

import os
import io
import json
import base64
import zipfile
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from datetime import timedelta
import httpx
from llm_responses import extract_text_from_image, extract_text_from_html
from utils import store_in_supabase
# -------------------- Configuration --------------------
# Load environment variables or hardcode for testing
load_dotenv()

SUPABASE_KEY=os.getenv("SUPABASE_KEY")
SUPABASE_URL=os.getenv("SUPABASE_URL")
UNIPILE_API_KEY = os.getenv("UNIPILE_API_KEY")
UNIPILE_BASE_URL = os.getenv("UNIPILE_BASE_URL")
BACKEND_URL = os.getenv("BACKEND_URL")
supabase: Client = create_client(SUPABASE_URL,SUPABASE_KEY)


app = FastAPI(title="Bill Processor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- API Routes --------------------


@app.post("/process-bill/")
async def process_bill(
    file: UploadFile = File(...),
    org_id: str = Form(...),
):
    """
    Handles uploaded files:
      - Images (.jpg, .jpeg, .png): Extracts data via GPT Vision.
      - HTML files: Extracts data via GPT text model.
      - ZIP files: Iterates through each file (image or HTML), extracts data, and stores each in Supabase.
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="File is required.")
        if not org_id:
            raise HTTPException(status_code=400, detail="Missing organization ID (org_id).")

        filename = file.filename.lower()
        content_type = file.content_type
        print(f"📁 Received file: {filename} ({content_type})")

        # === Case 1: IMAGE FILE ===
        if content_type.startswith("image/") or filename.endswith((".jpg", ".jpeg", ".png")):
            image_bytes = await file.read()
            encoded = base64.b64encode(image_bytes)
            print("🧠 Sending image to GPT-4 Vision...")
            extracted_data = extract_text_from_image(encoded)
            extracted_data["file_type"] = "image"
            extracted_data["org_id"] = org_id

            print("💾 Storing extracted image data in Supabase...")
            stored_data = store_in_supabase(extracted_data)
            print("✅ Image data stored successfully:", stored_data)

            return JSONResponse(content={"status": "success", "data": stored_data})

        # === Case 2: HTML FILE ===
        elif filename.endswith(".html"):
            html_bytes = await file.read()
            html_content = html_bytes.decode("utf-8", errors="ignore")
            print(f"📄 HTML file received ({len(html_bytes)} bytes) — sending to GPT-4...")
            extracted_data = extract_text_from_html(html_content)
            extracted_data["file_type"] = "html"
            extracted_data["org_id"] = org_id

            print("💾 Storing extracted HTML data in Supabase...")
            stored_data = store_in_supabase(extracted_data)
            print("✅ HTML data stored successfully:", stored_data)

            return JSONResponse(content={"status": "success", "data": stored_data})

        # === Case 3: ZIP FILE ===
        elif filename.endswith(".zip"):
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, filename)

            # Save ZIP temporarily
            with open(zip_path, "wb") as buffer:
                buffer.write(await file.read())

            print(f"🗜️ ZIP saved to: {zip_path}")
            results = []
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
                file_list = zip_ref.namelist()
                print(f"📦 Extracted files: {file_list}")

                # Loop through each file
                for inner_file in file_list:
                    try:
                        file_path = os.path.join(temp_dir, inner_file)
                        ext = os.path.splitext(inner_file)[1].lower()

                        if ext in [".jpg", ".jpeg", ".png"]:
                            with open(file_path, "rb") as f:
                                encoded = base64.b64encode(f.read())
                            print(f"🖼️ Processing image inside ZIP: {inner_file}")
                            data = extract_text_from_image(encoded)
                            data["file_type"] = "image"

                        elif ext == ".html":
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                html_content = f.read()
                            print(f"🌐 Processing HTML inside ZIP: {inner_file}")
                            data = extract_text_from_html(html_content)
                            data["file_type"] = "html"

                        else:
                            print(f"⚠️ Skipping unsupported file: {inner_file}")
                            continue

                        # Add org_id + source filename
                        data["org_id"] = org_id
                        data["source_file"] = inner_file

                        # Store each file's data
                        stored = store_in_supabase(data)
                        results.append(stored)
                        print(f"✅ Stored data for: {inner_file}")

                    except Exception as inner_e:
                        print(f"❌ Error processing {inner_file}: {inner_e}")

            print(f"🏁 Finished ZIP processing. {len(results)} files processed.")
            return JSONResponse(
                content={
                    "status": "success",
                    "message": f"Processed {len(results)} supported files successfully.",
                    "data": results,
                }
            )

        # === Case 4: UNSUPPORTED FILE TYPE ===
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image, HTML, or ZIP.")

    except Exception as e:
        print("❌ Error in /process-bill/:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/{org_id}")
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
            bill["total_amount"] = float(bill.get("total_amount") or 0.0)

        return {"success": True, "data": bills}

    except Exception as e:
        print("Error fetching bills:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-auth-link/")
async def generate_auth_link(org_id: str = Form(...)):
    """
    Generates a Unipile WhatsApp authentication link for a specific org_id.
    Requires org_id from frontend.
    """
    try: 
        if not org_id:
            raise HTTPException(status_code=400, detail="Missing org_id")
        # Expiry time for the link
        expires_on = (datetime.utcnow() + timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        headers = {"accept": "application/json", "content-type": "application/json", "X-API-KEY": UNIPILE_API_KEY}

        payload = {
            "type": "create",
            "providers": ["WHATSAPP"],
            "expiresOn": expires_on,
            "api_url": UNIPILE_BASE_URL,
            "bypass_success_screen": False,
            "notify_url": f"{BACKEND_URL}/whatsapp/callback", 
            "name": org_id,  # 👈 webhook target                                  # 👈 echo back your user
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{UNIPILE_BASE_URL}/api/v1/hosted/accounts/link", json=payload, headers=headers)

        if response.status_code // 100 != 2:
            return JSONResponse(content={"error": response.text}, status_code=response.status_code)

        result = response.json()
        return JSONResponse(content={"url": result.get("url")}, status_code=200)
    except HTTPException as he:
        return JSONResponse(content={"error": he.detail}, status_code=he.status_code)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/whatsapp/callback")
async def whatsapp_callback(request: Request):
    """
    Webhook endpoint that receives notifications from Unipile 
    when WhatsApp account status changes
    """
    try:
        data = await request.json()

        # Extract key information from the webhook payload
        account_id = data.get("account_id")
        status = data.get("status")        # e.g., "connected", "disconnected", "error"
        provider = data.get("provider")    # Should be "WHATSAPP"
        org_id = data.get("name")          # The org_id passed in your setup

        print(f"📩 Webhook received - Account: {account_id}, Status: {status}, Org: {org_id}")

        # ✅ Upsert WhatsApp connection info
        res = supabase.table("whatsapp_connections").upsert(
            {
                "org_id": org_id,
                "account_id": account_id,
                "status": status,
                "provider": provider,
                "last_updated_at": datetime.utcnow().isoformat()
            },
            on_conflict="org_id"
        ).execute()

        print(f"🟢 Supabase upsert result: {res.data}")

        # ✅ Handle status cases (optional — you can add DB actions here)
        if status == "connected":
            print(f"✅ WhatsApp connected successfully for org: {org_id}")

        elif status == "disconnected":
            print(f"⚠️ WhatsApp disconnected for org: {org_id}")

        elif status == "error":
            error_message = data.get("error_message", "Unknown error")
            print(f"❌ WhatsApp connection error for org {org_id}: {error_message}")

        # ✅ Always return 200 to acknowledge webhook
        return JSONResponse(
            content={
                "status": "received",
                "account_id": account_id,
                "org_id": org_id,
                "processed": True
            },
            status_code=200
        )

    except Exception as e:
        print(f"🚨 Error processing webhook: {str(e)}")
        # Return 200 even on error so Unipile doesn’t retry
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=200
        )
# --- ROOT ENDPOINT ---
@app.get("/")
async def root():
    return {"message": "Enhanced Bill OCR API Running"}

# --- APP RUNNER ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


