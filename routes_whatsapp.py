import os
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Request
from pydantic import BaseModel
import httpx
import requests
from collections import defaultdict
from dotenv import load_dotenv

from supabase import create_client, Client
from llm_responses import generate_followup_message
from utils import update_contact_status, format_phone_number
# ============= CONFIG =============
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
WHATSAPP_SERVICE_URL = os.getenv("WHATSAPP_SERVICE_URL", "http://localhost:3001")




if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
router = APIRouter(prefix="/whatsapp", tags=["WhatsApp"])

# ============= IN-MEMORY STORAGE =============
active_sessions: Dict[str, dict] = {}
websocket_connections: dict[str, set[WebSocket]] = defaultdict(set)

# ============= MODELS =============
class StartSessionRequest(BaseModel):
    org_id: str



class WhatsAppWebhook(BaseModel):
    session_id: str
    event: str
    data: dict

# ============= WEBSOCKET HELPER =============
"""
async def broadcast_to_websockets(message: dict):
    Send message to all connected WebSocket clients
    dead_connections = set()
    for ws in websocket_connections:
        try:
            await ws.send_json(message)
        except:
            dead_connections.add(ws)
    websocket_connections.difference_update(dead_connections)
"""

async def broadcast_to_org(org_id: str, message: dict):
    """
    Send message ONLY to WebSocket clients of a specific org_id
    """
    if org_id not in websocket_connections:
        return

    dead_ws = set()
    for ws in websocket_connections[org_id].copy():  # .copy() to avoid mutation during iteration
        try:
            await ws.send_json(message)
        except Exception:
            dead_ws.add(ws)

    # Clean up dead connections
    for ws in dead_ws:
        websocket_connections[org_id].discard(ws)

    print(f"Broadcasted to org_id {org_id}: {message.get('type')}")


# ============= DATABASE HELPERS =============
def save_session_to_db(session_id: str, org_id: str, status: str, phone: str = None):
    """Save or update session in Supabase"""
    try:
        existing = supabase.table("whatsapp_sessions").select("id").eq("id", session_id).execute()
        data = {
            "id": session_id,
            "user_id": org_id,
            "qr_code": None,              
            "qr_expires_at": None,
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        if phone:
            data["phone_number"] = phone
        if existing.data:
            supabase.table("whatsapp_sessions").update(data).eq("id", session_id).execute()
        else:
            data["created_at"] = datetime.utcnow().isoformat()
            supabase.table("whatsapp_sessions").insert(data).execute()
    except Exception as e:
        print(f"DB Error: {e}")


# ============= ENDPOINTS =============

@router.post("/session/start")
async def start_session(request: StartSessionRequest):
    """Start a new WhatsApp session and generate QR"""
    session_id = request.org_id
    print(f"Starting WhatsApp session for org_id: {session_id}")
    async with httpx.AsyncClient() as client:
        try:
            url = f"{WHATSAPP_SERVICE_URL}/session/start"  # fixed
            print("➡️ Requesting:", url)
            response = await client.post(
                f"{WHATSAPP_SERVICE_URL}/session/start",
                json={"session_id": session_id},
                timeout=10.0
            )
            response.raise_for_status()
            print("Node.js service response:", response.status_code, await response.aread())
        except Exception as e:
            print(f"Error starting WA session: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start WA session: {str(e)}")
    active_sessions[session_id] = {
        "user_id": request.org_id,
        "status": "initializing",
        "qr": None,
        "phone": None,
        "created_at": datetime.utcnow()
    }
    save_session_to_db(session_id, request.org_id, "initializing")

    return {
        "session_id": session_id,
        "status": "initializing",
        "message": "Session started. Waiting for QR code..."
    }

@router.get("/session/{session_id}/status")

async def get_session_status(session_id: str):
    print(f"Fetching status for session: {session_id}")
    if session_id in active_sessions:
        return active_sessions[session_id]
    try:
        print("Querying Supabase for session status...")
        result = supabase.table("whatsapp_sessions").select("*").eq("id", session_id).single().execute()
        
        if result.data:
            return {
                "session_id": session_id,
                "status": result.data.get("status"),
                "qr": result.data.get("qr_code"),
                "phone": result.data.get("phone_number")
            }
        
    except:
        pass
    raise HTTPException(status_code=404, detail="Session not found")

from datetime import datetime
def fetch_customer_details(org_id: str, phone: str):
    """
    Fetch customer details from bills table.
    Returns dict with name, bill_date, and total_amount or None if not found.
    Automatically adds '91' prefix if number is 10 digits.
    """
    try:
        print(f"🔍 Fetching customer details for phone: {phone}, org_id: {org_id}")


        # 🧾 Query latest bill for org_id + phone
        res = (
            supabase.table("bills")
            .select("name, bill_date, total_amount")
            .eq("org_id", org_id)
            .eq("contact_number", phone)
            .order("bill_date", desc=True)
            .limit(1)
            .execute()
        )
        print("🧾 Supabase bills query result:", res.data)
        if res.data and len(res.data) > 0:
            customer = res.data[0]
            name = customer.get("name")
            bill_amount = customer.get("total_amount")
            bill_date = customer.get("bill_date")

            print(f"✅ Found customer: {name}, Bill: {bill_amount}, Date: {bill_date}")
            return {
                "name": name,
                "bill_date": bill_date,
                "bill_amount": float(bill_amount) if bill_amount else None,
            }

        print("⚠️ No bill found for this customer")
        return None

    except Exception as e:
        print(f"❌ Error fetching customer details: {str(e)}")
        return None


def store_message(session_id, phone, message, sender, org_id=None):
    """
    Store a WhatsApp message in Supabase.
    Upsert based on (session_id, phone).
    If it's a new conversation and org_id is provided, fetch customer details from bills table.
    """
    try:
        print(f"\n💾 Storing message for {phone} (Session: {session_id})")

        new_message = {sender: message}

        existing = (
            supabase.table("conversations")
            .select("message_list")
            .eq("session_id", session_id)
            .eq("phone", phone)
            .execute()
        )

        if existing.data and len(existing.data) > 0:
            conversation = existing.data[0].get("message_list", []) or []
            conversation.append(new_message)
            upsert_data = {
                "session_id": session_id,
                "phone": phone,
                "message_list": conversation,
            }
        else:
            conversation = [new_message]
            upsert_data = {
                "session_id": session_id,
                "phone": phone,
                "message_list": conversation,
                "name": None,
                "bill_amount": None,
                "bill_date": None,
            }

            if org_id:
                print(f"🔎 Fetching customer details for org_id={org_id}")
                customer_details = fetch_customer_details(org_id, phone)
                if customer_details:
                    upsert_data.update({
                        "name": customer_details.get("name"),
                        "bill_amount": customer_details.get("bill_amount"),
                        "bill_date": customer_details.get("bill_date"),
                    })
                    print(f"📋 Added customer details: {customer_details}")

        print("🧾 Upserting data:", upsert_data)

        res = (
            supabase.table("conversations")
            .upsert(upsert_data, on_conflict="session_id, phone")
            .execute()
        )

        print("✅ Message stored (upserted) successfully.")
        return res.data

    except Exception as e:
        print(f"❌ Error storing message: {e}")
        return None

@router.post("/send-whatsapp")
async def send_whatsapp(request: Request):
    """
    Sends WhatsApp message with a retry mechanism that mimics 
    'clicking the button again' if the first attempt fails.
    """
    
    # 🛡️ MASTER TRY/CATCH BLOCK
    try:
        body = await request.json()
        print("\n📩 Incoming Request Body:", body)
        
        org_id = body.get("org_id")
        phone = body.get("phone")
        text = body.get("message")
        
        print(f"➡️ Preparing to send message to phone: {phone} for org_id: {org_id}")

        # 1. Validate Inputs
        if not org_id or not phone or not text:
            raise HTTPException(status_code=400, detail="org_id, phone, and message are required")

        sender = "res_owner"
        
        # 2. Check Session Status
        res = supabase.table("whatsapp_sessions").select("id, status").eq("id", org_id).execute()
        
        if not res.data:
            raise HTTPException(status_code=404, detail="No WhatsApp session found for this org_id")
        
        session_info = res.data[0]
        session_id = session_info.get("id")
        status = session_info.get("status")
        
        if status != "connected":
            raise HTTPException(status_code=400, detail="WhatsApp session is not connected")

        # 3. Format Phone Number
        try:
            formatted_phone = format_phone_number(org_id, phone)
        except Exception as format_error:
            raise HTTPException(status_code=400, detail=f"Invalid phone format: {str(format_error)}")

        print(f"🚀 Connecting to Node.js service...")
        
        # ============================================================
        # 🔍 HEALTH CHECK — Wake up the Node.js WhatsApp Service
        # ============================================================
        async with httpx.AsyncClient() as client:
            try:
                print("🩺 Pinging WhatsApp service /health ...")

                health_res = await client.get(f"{WHATSAPP_SERVICE_URL}/health", timeout=5.0)

                print(f"🩺 Health response status: {health_res.status_code}")
                print(f"🩺 Health response body: {health_res.text}")

                if health_res.status_code != 200:
                    print("❌ WhatsApp service returned non-200 on /health")
                    raise HTTPException(status_code=503, detail="WhatsApp service unhealthy")

                print("✅ WhatsApp service is healthy — continuing to send message.")

            except Exception as e:
                print(f"❌ Health check failed: {e}")
                raise HTTPException(status_code=503, detail="WhatsApp service is not responding")

        # ============================================================
        # 🔄 RETRY LOGIC (Simulates 'Clicking Again')
        # ============================================================
        max_retries = 3
        last_exception = None
        response_data = None

        async with httpx.AsyncClient() as client:
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"🔹 Attempt {attempt}/{max_retries}...")
                    
                    response = await client.post(
                        f"{WHATSAPP_SERVICE_URL}/session/{session_id}/send",
                        json={"to": formatted_phone, "text": text},
                        timeout=30.0 
                    )
                    
                    response_text = await response.aread()
                    
                    # If we get a 4xx error (like 'Invalid Number'), retrying won't help. Stop immediately.
                    if 400 <= response.status_code < 500:
                        print(f"❌ Client Error {response.status_code}: {response_text}")
                        response.raise_for_status()

                    # If we get a 200 OK, we are done!
                    response.raise_for_status()

                    print(f"✅ Success on attempt {attempt}. Node Response: {response.status_code}")
                    response_data = response.json()
                    break # Exit the retry loop immediately

                except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
                    # This block catches Timeouts, Connection Refused, or 500 Errors
                    last_exception = e
                    print(f"⚠️ Attempt {attempt} failed: {str(e)}")
                    
                    if attempt < max_retries:
                        # 👉 THIS IS THE FIX: Wait 2 seconds before the 'next req'
                        print("⏳ Waiting 2 seconds before retrying (waking up connection)...")
                        await asyncio.sleep(2)
                    else:
                        print("❌ All retries exhausted.")

        # 4. Final Check
        if response_data is None:
            print("❌ Final Failure: Could not send message after retries.")
            if isinstance(last_exception, httpx.TimeoutException):
                raise HTTPException(status_code=504, detail="Node.js service timed out (Server might be restarting)")
            elif last_exception:
                raise HTTPException(status_code=502, detail=f"Failed to send after {max_retries} attempts: {str(last_exception)}")
            else:
                raise HTTPException(status_code=500, detail="Unknown error during retries")

        # ============================================================
        # 💾 DATABASE STORAGE (Only happens if loop succeeds)
        # ============================================================
        print("💾 Storing message in Supabase...")
        store_message(session_id, formatted_phone, text, sender, org_id)
        update_contact_status(org_id, phone)
        
        return {"success": True, "data": response_data}

    # 🚨 CATCH KNOWN HTTP EXCEPTIONS
    except HTTPException as he:
        raise he

    # 🚨 CATCH EVERYTHING ELSE (Crash Report)
    except Exception as e:
        print("\n🛑 CRITICAL INTERNAL SERVER ERROR 🛑")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/session/{session_id}/messages")
async def get_messages(session_id: str, limit: int = 50):
    """Get messages for a session"""
    try:
        result = (
            supabase.table("messages")
            .select("*")
            .eq("session_id", session_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/webhook/whatsapp")
async def whatsapp_webhook_handler(webhook: WhatsAppWebhook):
    """
    Generic webhook handler for Node.js events.
    Node.js ALWAYS posts to: /webhook/whatsapp
    """
    print(f"📩 General WhatsApp webhook: {webhook.event}")

    if webhook.event == "qr_ready":
        return await whatsapp_qr(webhook)

    if webhook.event == "connected":
        return await whatsapp_connected(webhook)

    if webhook.event == "disconnected":
        return await whatsapp_disconnected(webhook)

    if webhook.event == "message_received":
        return await whatsapp_message(webhook)

    if webhook.event == "user_logout":
        return await whatsapp_user_logout(webhook)

    print(f"⚠️ Unknown event: {webhook.event}")
    return {"success": False, "error": "Unknown event"}

@router.post("/session/{session_id}/disconnect")
async def disconnect_whatsapp(session_id: str):
    """
    Proper WhatsApp disconnect:
    1. Tell Node to remove session from memory
    2. Clear Supabase auth_data & status
    """
    print(f"🔹 Disconnecting WhatsApp session: {session_id}")

    # 1️⃣ Tell Node.js to disconnect the WhatsApp socket
    try:
        async with httpx.AsyncClient() as client:
            node_res = await client.post(
                f"{WHATSAPP_SERVICE_URL}/session/{session_id}/disconnect",
                timeout=10.0
            )
            print("🔌 Node disconnect:", node_res.status_code, node_res.text)
    except Exception as err:
        print(f"⚠️ Node disconnect failed (not fatal): {err}")

    # 2️⃣ Clear Supabase session (auth_data + status)
    try:
        update_res = supabase.table("whatsapp_sessions").update(
            {
                "status": "disconnected",
                "auth_data": None,
                "qr_code": None,
                "phone_number": None,
                "updated_at": datetime.utcnow().isoformat()
            }
        ).eq("id", session_id).execute()

        print(f"🧹 Supabase updated:", update_res.data)

    except Exception as e:
        print(f"❌ Supabase update error:", e)
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "message": "WhatsApp session disconnected successfully",
        "session_id": session_id
    }

# ============= WEBHOOK =============
# ============= WEBHOOK (MULTI-ENDPOINT VERSION) =============
from fastapi import BackgroundTasks

# ✅ QR READY EVENT
@router.post("/webhook/qr")
async def whatsapp_qr(webhook: WhatsAppWebhook):
    """Handle QR code generation events"""
    session_id = webhook.session_id
    data = webhook.data
    print(f"📥 [QR] Webhook received for session {session_id}")

    if session_id not in active_sessions:
        active_sessions[session_id] = {
            "user_id": session_id,
            "status": "qr_ready"
        }
    else:
        active_sessions[session_id].update({"status": "qr_ready"})

    expires_in = data.get("expires_in", 60)
    expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
    print("ATTEMPTING STORAGE TO SUPABASE!!!!")

    supabase.table("whatsapp_sessions").update({
        "status": "qr_ready",
        "qr_code": data.get("qr"),
        "qr_expires_at": expires_at.isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }).eq("id", session_id).execute()
    print("STORED IN SUPABASE!!!!")

    await broadcast_to_org(session_id, {
        "type": "qr_update",
        "session_id": session_id,
        "qr": data.get("qr"),
        "expires_in": data.get("expires_in", 60)
    })

    print(f"✅ QR event processed for {session_id}")
    return {"success": True}


# ✅ CONNECTED EVENT
@router.post("/webhook/connected")
async def whatsapp_connected(webhook: WhatsAppWebhook):
    """Handle WhatsApp connection established events"""
    session_id = webhook.session_id
    data = webhook.data
    phone = data.get("phone")
    name = data.get("name", "User")

    print(f"✅ [Connected] {session_id} — {phone} ({name})")

    if session_id not in active_sessions:
        active_sessions[session_id] = {"status": "connected", "phone": phone}
    else:
        active_sessions[session_id].update({"status": "connected", "phone": phone})

    save_session_to_db(session_id, active_sessions[session_id].get("user_id", "session_id"), "connected", phone=phone)

    await broadcast_to_org(session_id, {
        "type": "connected",
        "session_id": session_id,
        "phone": phone,
        "name": name
    })

    return {"success": True}


# ✅ DISCONNECTED EVENT
@router.post("/webhook/disconnect")
async def whatsapp_disconnected(webhook: WhatsAppWebhook):
    """Handle WhatsApp disconnection events"""
    session_id = webhook.session_id
    org_id = session_id
    reason = webhook.data.get("reason", "unknown")

    print(f"⚠️ [Disconnected] {session_id} — Reason: {reason}")

    if session_id in active_sessions:
        del active_sessions[session_id]

    save_session_to_db(session_id, org_id, "disconnected")

    await broadcast_to_org(session_id, {
        "type": "disconnected",
        "session_id": session_id,
        "reason": reason
    })

    print(f"✅ Disconnected event processed for {session_id}")
    return {"success": True}


def format_whatsapp_message(llm_response: str) -> str:
    """
    Dynamically format LLM response for WhatsApp readability.
    Breaks long text into chunks, adds line breaks, preserves links.
    """
    import re
    
    # Split into sentences while preserving links
    sentences = re.split(r'(?<=[.!?])\s+', llm_response)
    
    formatted_lines = []
    buffer = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If buffer + sentence > 100 chars, flush buffer
        if len(buffer) + len(sentence) > 100 and buffer:
            formatted_lines.append(buffer)
            buffer = sentence
        else:
            buffer += " " + sentence if buffer else sentence
    
    if buffer:
        formatted_lines.append(buffer)
    
    # Join with double line breaks
    formatted = "\n\n".join(formatted_lines)
    
    return formatted


# ✅ MESSAGE RECEIVED EVENT
@router.post("/webhook/message")
async def whatsapp_message(webhook: WhatsAppWebhook, background_tasks: BackgroundTasks):
    """Handle incoming WhatsApp messages"""
    session_id = webhook.session_id
    data = webhook.data

    print(f"\n💬 [Message] Webhook received for session: {session_id}")
    print(f"🔹 Data: {data}")


    remote_jid = data.get("remoteJid", "")
    phone = None
    
    # Check if it's a LID (LinkedIn Identity Device / Channel)
    if "@lid" in remote_jid:
        print("🔹 LID detected — extracting from senderPn")
        # Extract from raw.key.senderPn
        raw_data = data.get("raw", {})
        key_data = raw_data.get("key", {})
        sender_pn = key_data.get("senderPn", "")
        
        if sender_pn and "@s.whatsapp.net" in sender_pn:
            phone = sender_pn.replace("@s.whatsapp.net", "").replace("+", "").strip()
            print(f"✅ Phone extracted from senderPn: {phone}")
        else:
            print("⚠️ No valid senderPn found — phone will be null")
            phone = None
    
    # Regular WhatsApp JID
    elif "@s.whatsapp.net" in remote_jid:
        phone = remote_jid.replace("@s.whatsapp.net", "").replace("+", "").strip()
        print(f"✅ Phone extracted from remoteJid: {phone}")
    
    # Group or unsupported JID
    else:
        print(f"⚠️ Unsupported JID format: {remote_jid} — skipping")
        return {"success": True}

    # If we still don't have a phone number, skip
    if not phone:
        print("⚠️ Could not extract phone number — skipping conversation lookup")
        return {"success": True}

    # 2️⃣ Extract message content
    message_obj = data.get("message", {})
    print(f"🔹 Message Object: {message_obj}")
    if isinstance(message_obj, dict):
        message_text = message_obj.get("text", "")
    elif isinstance(message_obj, str):
        message_text = message_obj
    else:
        message_text = ""

    is_from_me = data.get("fromMe", False)
    sender = "res_owner" if is_from_me else "res_customer"

    print(f"👤 Sender: {sender} | 💬 Message: {message_text}")

    # 3️⃣ Check if conversation exists
    try:
        conv = supabase.table("conversations") \
            .select("message_list") \
            .eq("session_id", session_id) \
            .eq("phone", int(phone)) \
            .maybe_single() \
            .execute()
    except Exception as e:
        print(f"DB error: {e}")
        return {"success": False, "error": "DB error"}

    # IF NO CONVERSATION EXISTS → IGNORE EVERYTHING (no store, no AI)
    if not conv:
        print(f"IGNORED message from {phone} → no conversation exists yet")
        return {"success": True}

    # IF CONVERSATION EXISTS → now we process normally
    print(f"Conversation exists for {phone} → storing message & triggering AI")

    # Store incoming message
    store_message(session_id, int(phone), message_text, "res_customer")

    # Update message list
    message_list = conv.data.get("message_list", [])
    message_list.append({"res_customer": message_text})

    # Trigger AI reply in background
    async def handle_ai_followup():
        try:
            print("🚀 Background task started")

            # 1️⃣ Store customer message
            print("💾 Saving customer message...")
            restaurant_name, google_review_link = "Rohan's Rest", "my-restaurant-link"
            print("🤖 Generating AI reply...")
            new_message = await generate_followup_message(message_list, restaurant_name, google_review_link)
            formatted = format_whatsapp_message(new_message)
            print("📨 AI message to be sent as the next step:", formatted)

            print("📡 Sending AI reply to WhatsApp...")
            max_retries=3
            async with httpx.AsyncClient() as client:
                for attempt in range(1, max_retries + 1):
                    try:
                        print(f"🔹 Attempt {attempt}/{max_retries}...")
                        
                        response=await client.post(
                                f"{WHATSAPP_SERVICE_URL}/session/{session_id}/send",
                                json={"to": phone, "text": formatted},
                                timeout=15.0
                        )
                        response_text = await response.aread()
                        
                        # If we get a 4xx error (like 'Invalid Number'), retrying won't help. Stop immediately.
                        if 400 <= response.status_code < 500:
                            print(f"❌ Client Error {response.status_code}: {response_text}")
                            response.raise_for_status()

                        # If we get a 200 OK, we are done!
                        response.raise_for_status()

                        print(f"✅ Success on attempt {attempt}. Node Response: {response.status_code}")
                        break # Exit the retry loop immediately

                    except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
                        # This block catches Timeouts, Connection Refused, or 500 Errors
                        print(f"⚠️ Attempt {attempt} failed: {str(e)}")
                        
                        if attempt < max_retries:
                            # 👉 THIS IS THE FIX: Wait 2 seconds before the 'next req'
                            print("⏳ Waiting 2 seconds before retrying (waking up connection)...")
                            await asyncio.sleep(2)
                        else:
                            print("❌ All retries exhausted.")

            print("💾 Saving AI reply in DB...")
            
            store_message(session_id, int(phone), formatted, "res_owner")
            
            await broadcast_to_org(session_id, {
                "type": "new_message",
                "session_id": session_id,
                "message": {"text": formatted, "sender": "res_owner"}
            })

            print("✅ Background task completed successfully")

        except Exception as e:
            print(f"AI reply error: {e}")

    background_tasks.add_task(handle_ai_followup)

    return {"success": True}

@router.post("/webhook/user-logout")
async def whatsapp_user_logout(request: Request):
    """Handles logout initiated directly from WhatsApp (device removed)."""
    body = await request.json()
    print("📡 [USER LOGOUT WEBHOOK] Received:", body)

    session_id = body.get("session_id")
    data = body.get("data", {})
    reason = data.get("reason", "unknown")
    message = data.get("message", "User logged out")

    try:
        # Update session status in database
        supabase.table("whatsapp_sessions")\
            .update({
                "user_id": "",
                "status": "logged_out",
                "auth_data": None,
                "phone_number": None
            })\
            .eq("id", session_id)\
            .execute()

        print(f"✅ Session {session_id} marked as logged out in database")

        # Broadcast to all connected WebSocket clients
        await broadcast_to_org(session_id, {
            "type": "user_logout",
            "session_id": session_id,
            "status": "logged_out",
            "reason": reason,
            "message": message,
            "timestamp": data.get("timestamp")
        })

        print(f"📢 Broadcasted logout notification to WebSocket clients")

        return {
            "received": True,
            "message": f"Session {session_id} logout processed successfully"
        }

    except Exception as e:
        print(f"❌ Error processing user logout: {e}")
        return {
            "received": False,
            "error": str(e)
        }
# ============= WEBSOCKET =============

@router.websocket("/ws/{org_id}")
async def websocket_endpoint(websocket: WebSocket, org_id: str):
    """WebSocket connection for real-time updates — one per restaurant/org"""
    print(f"New WebSocket connection request for org_id: {org_id}")
    await websocket.accept()
    print(f"WebSocket accepted for org_id: {org_id}")

    # Add this specific restaurant's connection
    websocket_connections[org_id].add(websocket)

    try:
        # Confirm connection
        await websocket.send_json({
            "type": "connection_ack",
            "message": "WebSocket connected",
            "org_id": org_id
        })

        # Keep-alive loop (optional pong)
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "pong", "data": data})

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for org_id: {org_id}")
        websocket_connections[org_id].discard(websocket)
    except Exception as e:
        print(f"WebSocket error for org_id {org_id}: {e}")
        websocket_connections[org_id].discard(websocket)

# ============= HEALTH CHECK =============
@router.get("/health")

async def health():
    print("🔍 Health check requested")
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "websocket_connections": len(websocket_connections),
    }