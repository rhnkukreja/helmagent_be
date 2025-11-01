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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
supabase: Client = create_client(SUPABASE_URL,SUPABASE_KEY)


import httpx
import json
from typing import List, Dict, Any
import asyncio

UNIPILE_API_KEY = "nnX2Om8V./30Vp9ruk3RXaLBxcydSAWXwJqantAMKf5goHMhLqUs="
UNIPILE_BASE_URL = "https://api13.unipile.com:14315"
BACKEND_URL = "https://add457c65882.ngrok-free.app"


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


from fastapi import Request
import traceback
import json





@app.post("/generate-auth-link/")
async def generate_auth_link(org_id: str = Form(...)):
    try:
        if not org_id:
            raise HTTPException(status_code=400, detail="Missing org_id")

        dt = datetime.utcnow() + timedelta(minutes=10)
        expires_on = dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{dt.microsecond // 1000:03d}Z"

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-API-KEY": UNIPILE_API_KEY,
        }

        payload = {
            "type": "create",
            "providers": ["WHATSAPP"],
            "expiresOn": expires_on,
            "api_url": UNIPILE_BASE_URL,  # ✅ THIS WAS MISSING
            "bypass_success_screen": False,
            "notify_url": f"{BACKEND_URL}/whatsapp/callback",
            "name": org_id,
        }

        print("=" * 50)
        print("PAYLOAD BEING SENT:")
        print(json.dumps(payload, indent=2))
        print("=" * 50)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{UNIPILE_BASE_URL}/api/v1/hosted/accounts/link",
                json=payload,
                headers=headers,
            )

        print("RESPONSE STATUS:", response.status_code)
        print("RESPONSE BODY:", response.text)

        if response.status_code // 100 != 2:
            return JSONResponse(
                content={"error": response.text},
                status_code=response.status_code,
            )

        result = response.json()
        auth_url = result.get("url")

        return JSONResponse(content={"url": auth_url}, status_code=200)

    except HTTPException as he:
        return JSONResponse(content={"error": he.detail}, status_code=he.status_code)
    except Exception as e:
        print("EXCEPTION:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)







@app.post("/whatsapp/callback")
async def whatsapp_callback(request: Request):
    """
    Webhook endpoint that receives notifications from Unipile 
    when WhatsApp account status changes and updates Supabase.
    """
    try:
        data = await request.json()

        # 🔹 Extract key fields
        account_id = data.get("account_id")
        status = data.get("status")        # e.g. "connected", "disconnected", "error"
        provider = data.get("provider")    # Should be "WHATSAPP"
        org_id = data.get("name")          # org_id passed during auth link generation

        print(f"📩 Webhook received - Account: {account_id}, Status: {status}, Org: {org_id}")

        # 🔹 Validate required fields
        if not org_id or not status:
            print("❌ Missing required fields in webhook data.")
            return JSONResponse(
                content={"status": "error", "message": "Missing org_id or status"},
                status_code=200  # Always return 200 to avoid Unipile retries
            )

        # 🔹 Upsert WhatsApp connection info into Supabase
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

        # 🔹 Handle specific statuses
        if status == "connected":
            print(f"✅ WhatsApp connected successfully for org: {org_id}")

        elif status == "disconnected":
            print(f"⚠️ WhatsApp disconnected for org: {org_id}")

        elif status == "error":
            error_message = data.get("error_message", "Unknown error")
            print(f"❌ WhatsApp connection error for org {org_id}: {error_message}")

        # 🔹 Always acknowledge webhook
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
        # Return 200 to stop Unipile from retrying
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=200
        )

@app.get("/whatsapp/status/{org_id}")
async def get_whatsapp_status(org_id: str):
    res = supabase.table("whatsapp_connections").select("*").eq("org_id", org_id).execute()
    
    if not res.data:
        return {"status": "not_connected"}
    return res.data[0]


from openai import OpenAI
from pydantic import BaseModel
from datetime import datetime

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv(OPENAI_API_KEY))

class GenerateMessageRequest(BaseModel):
    name: str
    phone: str
    items_ordered: str
    order_date: str
    total_amount: float


@app.post("/generate-message")
async def generate_message(request: GenerateMessageRequest):
    """
    Generate personalized WhatsApp message using AI based on customer order data
    """
    try:
        print("\n" + "="*60)
        print("🤖 GENERATING AI MESSAGE")
        print("="*60)
        print(f"Customer: {request.name}")
        print(f"Phone: {request.phone}")
        print(f"Items: {request.items_ordered}")
        print(f"Date: {request.order_date}")
        print(f"Amount: ₹{request.total_amount}")
        
        # Create AI prompt
        prompt = f"""Generate a friendly, personalized WhatsApp message for a restaurant customer asking for feedback.

Customer Details:
- Name: {request.name}
- Order Date: {request.order_date}
- Items Ordered: {request.items_ordered}
- Total Amount: ₹{request.total_amount}

Requirements:
1. Use casual, friendly tone with emojis
2. Thank them for visiting
3. Mention 1-2 specific items they ordered and tell them facts about it , like chef's special, specially sourced ingredient etc
4. Ask how their experience was
5. Encourage them to share feedback
6. Keep it under 150 words
7. End with a call-to-action

Restaurant Name: The Corner Cafe

Generate the message now and just return the message:"""

        # Call OpenAI API
        # response = client.chat.completions.create(
        #     model="gpt-4o-mini",  # or gpt-4o for better quality
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You are a friendly restaurant manager writing personalized WhatsApp messages to customers. Be warm, genuine, and encourage honest feedback."
        #         },
        #         {
        #             "role": "user",
        #             "content": prompt
        #         }
        #     ],
        #     temperature=0.7,
        #     max_tokens=300
        # )
        
        # ai_message = response.choices[0].message.content.strip()
        ai_message="jbkjbjb"
        
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



async def generate_followup_message(message_list):
    """
    Generate a contextual WhatsApp follow-up message based on previous AI messages and customer order data
    """
    try:
        prompt = f"""You are a restaurant manager sending a friendly WhatsApp follow-up message to a returning or past customer.

Previous Messages:
{message_list}

Requirements:
1. Start with a friendly greeting using the customer’s name (use emojis naturally)
2. Refer to their previous visit casually (e.g., “Hope you enjoyed your Steam Rice last time!”)
3. Offer a light incentive, new dish recommendation, or invite them again (depending on tone)
4. Keep tone warm, conversational, and under 120 words
5. Include restaurant name: The Corner Cafe
6. End with a call to action (like “We’d love to host you again this weekend!”)
7. Don’t repeat the same text as previous messages — make it feel like a real follow-up.

Generate the follow-up WhatsApp message now and just return the message:"""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly restaurant manager following up with customers on WhatsApp. Write human-like, engaging, and caring messages that sound personal."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8,
            max_tokens=300
        )

        ai_message = response.choices[0].message.content.strip()

        print(f"\n✅ Follow-up AI Message Generated:")
        print(ai_message)
        print("="*60 + "\n")

        return ai_message

    except Exception as e:
        print(f"\n❌ Error generating follow-up message: {str(e)}")
        return JSONResponse(
            content={
                "success": False,
                "error": f"Failed to generate follow-up message: {str(e)}"
            },
            status_code=500
        )




from datetime import datetime

def store_to_supabase(account_id, phone, message, sender, chat_id=None):
    """
    Store a WhatsApp message in Supabase.
    If (account_id, phone) exists -> append message to conversation list
    Else -> create a new chat row
    """

    try:
        print(f"\n💾 Storing message for {phone} (Account: {account_id})")

        # Step 1: Check if chat already exists
        existing = supabase.table("conversations") \
            .select("*") \
            .eq("account_id", account_id) \
            .eq("phone", phone) \
            .execute()


        if existing.data and len(existing.data) > 0:
            # Chat exists — append to conversation list
            chat_id = existing.data[0]["chat_id"]
            conversation = existing.data[0].get("message_list", []) or []
            message_json_to_append={sender:message}
            conversation.append(message_json_to_append)

            res = supabase.table("conversations") \
                .update({"message_list": conversation}) \
                .eq("chat_id", chat_id) \
                .execute()

            print("🟢 Message appended to existing chat.")
            return res.data

        else:
            # Chat does not exist — create new row
            new_chat = {
                "account_id": account_id,
                "chat_id": chat_id,
                "phone": phone,
                "message_list": [{"res_owner":message}],
                "created_at": datetime.utcnow().isoformat(),
                "bill_price": None,
                "order_date": None,
            }

            res = supabase.table("conversations").insert(new_chat).execute()
            print("🆕 New chat created in Supabase.")
            return res.data

    except Exception as e:
        print(f"❌ Error storing message: {e}")
        return None




import requests




def get_chat_id(account_id: str, phone: str):
    """
    Fetch Unipile chat_id for a given phone number.
    Returns chat_id if found, else None.
    """
    try:
        url = f"{UNIPILE_BASE_URL}/api/v1/chats"
        params = {
            "limit": 10,
            "account_type": "WHATSAPP",
            "account_id": account_id
        }
        headers = {
            "accept": "application/json",
            "X-API-KEY": UNIPILE_API_KEY
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        formatted_phone = f"91{phone}@s.whatsapp.net"
        data = response.json()

        print(len(data["items"]))

        for item in data["items"]:
            print(item["provider_id"])
            if item["provider_id"]==formatted_phone:
                chat_id=item["id"]
                return chat_id
            else:
                print(f"❌ No chat found for phone: {phone}")



    except Exception as e:
        print(f"❌ Error fetching chat_id: {e}")
        return None




@app.post("/send-whatsapp")
async def send_whatsapp(request: Request):
    """
    Send WhatsApp message to a single customer using Unipile
    """
    try:
        print("\n" + "="*60)
        print("📤 SENDING WHATSAPP MESSAGE")
        print("="*60)
        

        # Parse request body
        body = await request.json()
        org_id= body.get("org_id")
        # phone = body.get("phone")
        phone='9805345415'
        message = body.get("message")

        
        if not phone or not message:
            raise HTTPException(status_code=400, detail="org_id,Phone and message are required")
        
        print(f"To: {phone}")
        print(f"Message: {message[:100]}...")
        res = supabase.table("whatsapp_connections").select("account_id").eq("org_id", org_id).execute()

        if not res.data or len(res.data) == 0 or not res.data[0].get("account_id"):
            raise HTTPException(status_code=404, detail="No WhatsApp account linked for this org_id")


        account_id = res.data[0]["account_id"]
        print(f"✅ Found account_id: {account_id}")
        # Convert to WhatsApp JID format
        formatted_phone = f"91{phone}@s.whatsapp.net"
        print(f"📱 Sending to WhatsApp ID: {formatted_phone}")
        print(f"account_id: {account_id}")


        # phone='+918219467323'
        # account_id = "HL9SULzcQ_qhWXOKWdnryQ"

        # Headers for Unipile
        headers = {
            "accept": "application/json",
            "content-type": "application/x-www-form-urlencoded",      
            "X-API-KEY": UNIPILE_API_KEY
        }
        form_data = {
            "account_id": account_id,
            "attendees_ids": formatted_phone,
            "text": message
        }
        
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            
            # Send message via Unipile

            # ✅ FIXED: Use 'data' without 'headers' parameter, or create new client with headers

            send_response = await client.post(
                f"{UNIPILE_BASE_URL}/api/v1/chats",
                data=form_data
            )

            
            print(f"Unipile Response: {send_response.status_code}")
            
            if send_response.status_code not in [200, 201]:
                error_detail = send_response.text
                print(f"❌ Failed: {error_detail}")
                raise HTTPException(
                    status_code=send_response.status_code,
                    detail=f"Failed to send WhatsApp message: {error_detail}"
                )
            
            result = send_response.json()
            print(f"✅ Message sent successfully!")
            print("="*60 + "\n")

            # fetch chat_id by listing all chats and finding the one with the specific phone number
            chat_id=get_chat_id(account_id, phone)

            # Store message to Supabase
            store_to_supabase(account_id, phone, message, sender="res-owner", chat_id=chat_id)


            return {
                "success": True,
                "message": "WhatsApp message sent successfully",
                "data": result
            }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )






async def fetch_chat_messages(chat_id: str) -> List[Dict[str, Any]]:
    """
    Fetch messages from a specific chat using Unipile API.
    Returns list of messages, most recent first.
    """
    headers = {
        "accept": "application/json",
        "X-API-KEY": UNIPILE_API_KEY
    }
    
    print(f"\n📥 Fetching messages for chat: {chat_id}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{UNIPILE_BASE_URL}/api/v1/chats/{chat_id}/messages",
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                # Extract the list of messages from 'items' key
                messages = data.get('items', [])
                print(f"✅ Fetched {len(messages)} messages")
                return messages        
            else:
                print(f"❌ Fetch error: {response.status_code} - {response.text}")
                raise Exception(f"Failed to fetch messages: {response.status_code}")

    except Exception as e:
        print(f"❌ Exception fetching messages: {str(e)}")
        raise


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


async def send_message_to_chat(account_id, phone, chat_id: str, message: str) -> Dict[str, Any]:
    """
    Send a message to a specific chat using Unipile API.
    """
    headers = {
        "accept": "application/json",
        "X-API-KEY": UNIPILE_API_KEY
        # content-type will be set to multipart/form-data by httpx when using data
    }
    
    form_data = {
        "text": message,
        "account_id":account_id
    }

    store_to_supabase(account_id, phone, message, sender="res-owner", chat_id=chat_id)

    
    print(f"\n📤 Sending message to chat: {chat_id}")
    print(f"Message: {message}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{UNIPILE_BASE_URL}/api/v1/chats/{chat_id}/messages",
                headers=headers,
                data=form_data
            )
        
        print(f"Response: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        if response.status_code in [200, 201]:
            print("\n✅ Message sent successfully!")
        else:
            print(f"\n❌ Failed with status {response.status_code}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Exception sending message: {str(e)}")
        return {"error": str(e)}


from fastapi import FastAPI, Request
from typing import Any, Dict


@app.post("/generate-send-followup")
async def generate_and_send_response(request: Request) -> Dict[str, Any]:
    """
    Webhook handler:
    - Accepts incoming WhatsApp webhook JSON.
    - Extracts account_id, chat_id, and message.
    - Validates chat_id against allowed list before proceeding.
    """
    try:
        # Parse the incoming JSON
        data = await request.json()
        print("Incoming webhook data:", data)

        # Extract required fields safely
        event_type = data.get("event")
        account_id = data.get("account_id")
        chat_id = data.get("chat_id")
        message = data.get("message")
        attendees_list = data.get("attendees")
        attendee_provider_id= attendees_list[0]["attendee_provider_id"]
        phone = attendee_provider_id.replace("@s.whatsapp.net", "").replace("+", "").strip()

        # ✅ Get sender info CORRECTLY
        sender_info = data.get("sender", {})
        sender_provider_id = sender_info.get("attendee_provider_id", "")
        sender_phone = sender_provider_id.replace("@s.whatsapp.net", "").replace("+", "").strip()

        # ✅ Check if bot sent this message
        if sender_phone in ["919857240000", "9857240000"]:
            print("🚫 Ignoring message from bot itself")
            return {"success": True, "message": "Ignored bot message"}

        store_to_supabase(account_id, phone, message, sender="res_customer", chat_id=chat_id)

        # 1. Fetch chats (messages)
        messages = await fetch_chat_messages(chat_id)
        message_list= [msg.get("text", "") for msg in messages if msg.get("text")]
        
        # 2. Generate new message using LLM
        new_message = await generate_followup_message(message_list)
        
        # 3. Send the message to the chat
        send_result = await send_message_to_chat(account_id, phone, chat_id, new_message)
        
        return {
            "success": True,
            "account_id": account_id,
            "chat_id": chat_id,
            "generated_message": new_message,
            "send_result": send_result
        }
        
    except Exception as e:
        print(f"\n❌ Overall error: {str(e)}")
        return {
            "success": False,
            "account_id": account_id,
            "chat_id": chat_id,
            "error": str(e)
        }


# account_id = "HL9SULzcQ_qhWXOKWdnryQ"  # From list_accounts
# chat_id = "upH0ipuLWw2EqU04JM2oUQ"  # Get from listing chats or webhook




# --- ROOT ENDPOINT ---
@app.get("/")
async def root():
    return {"message": "Enhanced Bill OCR API Running"}

# --- APP RUNNER ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



