import os
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Request
from pydantic import BaseModel
import httpx
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from llm_responses import generate_followup_message
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
websocket_connections: Set[WebSocket] = set()

# ============= MODELS =============
class StartSessionRequest(BaseModel):
    org_id: str



class WhatsAppWebhook(BaseModel):
    session_id: str
    event: str
    data: dict

# ============= WEBSOCKET HELPER =============
async def broadcast_to_websockets(message: dict):
    """Send message to all connected WebSocket clients"""
    dead_connections = set()
    for ws in websocket_connections:
        try:
            await ws.send_json(message)
        except:
            dead_connections.add(ws)
    websocket_connections.difference_update(dead_connections)

# ============= DATABASE HELPERS =============
def save_session_to_db(session_id: str, org_id: str, status: str, qr: str = None, phone: str = None):
    """Save or update session in Supabase"""
    try:
        existing = supabase.table("whatsapp_sessions").select("id").eq("id", session_id).execute()
        data = {
            "id": session_id,
            "user_id": org_id,
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        if qr:
            data["qr_code"] = qr
        if phone:
            data["phone_number"] = phone
        if existing.data:
            supabase.table("whatsapp_sessions").update(data).eq("id", session_id).execute()
        else:
            data["created_at"] = datetime.utcnow().isoformat()
            supabase.table("whatsapp_sessions").insert(data).execute()
    except Exception as e:
        print(f"DB Error: {e}")

def save_message_to_db(session_id: str, msg_data: dict):
    """Save message to Supabase"""
    try:
        supabase.table("messages").insert({
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "whatsapp_message_id": msg_data.get("id"),
            "from_number": msg_data.get("from"),
            "is_from_me": msg_data.get("fromMe", False),
            "message_type": msg_data.get("message", {}).get("type", "text"),
            "message_text": msg_data.get("message", {}).get("text", ""),
            "timestamp": datetime.fromtimestamp(int(msg_data.get("timestamp", 0))).isoformat(),
            "created_at": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        print(f"DB Error saving message: {e}")

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
    """Send WhatsApp message using org_id (same as session id)"""
    body = await request.json()
    print("\n📩 Incoming Request Body:", body)
    org_id = body.get("org_id")
    phone = body.get("phone")
    text = body.get("message")
    print(f"➡️ Preparing to send message to phone: {phone} for org_id: {org_id}")

    if not org_id or not phone or not text:
        raise HTTPException(status_code=400, detail="org_id, phone, and message are required")

    sender = "res_owner"
    print(f"➡️ Checking session in Supabase for org_id: {org_id}")

    res = supabase.table("whatsapp_sessions").select("id, status").eq("id", org_id).execute()
    print("🧾 Supabase Result:", res.data)

    if not res.data:
        raise HTTPException(status_code=404, detail="No WhatsApp session found for this org_id")
    print("✅ Session found in Supabase.")
    session_info = res.data[0]
    print("➡️ Session Info:", session_info)
    session_id = session_info.get("id")
    print(f"➡️ Session ID: {session_id}")
    status = session_info.get("status")
    print(f"➡️ Session status: {status}")
    if status != "connected":
        raise HTTPException(status_code=400, detail="WhatsApp session is not connected")

    formatted_phone = str(phone).strip()
    if not formatted_phone.startswith("+91") and not formatted_phone.startswith("91"):
        formatted_phone = f"91{formatted_phone}"
    formatted_phone = formatted_phone.lstrip("+")

    print(f"🚀 Sending message to Node.js service for phone: {formatted_phone}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{WHATSAPP_SERVICE_URL}/session/{session_id}/send",
                json={"to": formatted_phone, "text": text},
                timeout=10.0
            )
            print("📤 Node.js Response:", response.status_code, await response.aread())
            response.raise_for_status()
            print("✅ Message sent successfully via Node.js service.")
            # ✅ Now pass org_id so we can fetch and store customer details
            store_message(session_id, phone, text, sender, org_id)

            return {"success": True, "data": response.json()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

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

@router.post("/session/{session_id}/disconnect")
async def disconnect_session(session_id: str):
    """
    Disconnect WhatsApp session and clean up credentials in Node.js + Supabase
    """
    print(f"🔌 Disconnecting WhatsApp session: {session_id}")

    try:
        async with httpx.AsyncClient() as client:
            # ✅ Ask Node.js service to disconnect & delete session folder
            print("➡️ Requesting Node.js service to disconnect session...")
            print(f"URL: {WHATSAPP_SERVICE_URL}/session/{session_id}/disconnect")
            response = await client.post(
                f"{WHATSAPP_SERVICE_URL}/session/{session_id}/disconnect",
                timeout=10.0
            )
            print("📤 Node.js disconnect response:", response.status_code, await response.aread())
            if response.status_code != 200:
                print(f"⚠️ Node.js disconnect failed: {response.text}")
    except Exception as e:
        print(f"❌ Error disconnecting Node.js session: {e}")

    # ✅ Remove session from memory
    if session_id in active_sessions:
        del active_sessions[session_id]
        print(f"🧹 Removed {session_id} from active_sessions")

    # ✅ Update Supabase to reflect disconnection
    try:
        save_session_to_db(session_id, "", "disconnected")
        print(f"✅ Updated Supabase session {session_id} to 'disconnected'")
    except Exception as e:
        print(f"⚠️ Failed to update Supabase: {e}")

    return {"success": True, "message": "Session disconnected and cleaned up"}


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
            "user_id": "unknown",
            "status": "qr_ready",
            "qr": data.get("qr")
        }
    else:
        active_sessions[session_id].update({"status": "qr_ready", "qr": data.get("qr")})

    save_session_to_db(session_id, active_sessions[session_id]["user_id"], "qr_ready", qr=data.get("qr"))

    await broadcast_to_websockets({
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
        active_sessions[session_id].update({"status": "connected", "phone": phone, "qr": None})

    save_session_to_db(session_id, active_sessions[session_id].get("user_id", ""), "connected", phone=phone)

    await broadcast_to_websockets({
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
    reason = webhook.data.get("reason", "unknown")

    print(f"⚠️ [Disconnected] {session_id} — Reason: {reason}")

    if session_id in active_sessions:
        del active_sessions[session_id]

    save_session_to_db(session_id, "", "disconnected")

    await broadcast_to_websockets({
        "type": "disconnected",
        "session_id": session_id,
        "reason": reason
    })

    print(f"✅ Disconnected event processed for {session_id}")
    return {"success": True}


# ✅ MESSAGE RECEIVED EVENT
@router.post("/webhook/message")
async def whatsapp_message(webhook: WhatsAppWebhook, background_tasks: BackgroundTasks):
    """Handle incoming WhatsApp messages"""
    session_id = webhook.session_id
    data = webhook.data

    print(f"\n💬 [Message] Webhook received for session: {session_id}")
    print(f"🔹 Data: {data}")

    # 1️⃣ Normalize phone number
    phone_raw = data.get("from", "")
    phone = phone_raw.replace("@s.whatsapp.net", "").replace("+", "").strip()
    if phone.startswith("91") and len(phone) > 10:
        phone = phone[-10:]
    phone_full = f"91{phone}" if not phone.startswith("91") else phone

    # 2️⃣ Extract message content
    message_obj = data.get("message", {})
    message_text = message_obj.get("text", "")
    is_from_me = data.get("fromMe", False)
    sender = "res_owner" if is_from_me else "res_customer"

    print(f"👤 Sender: {sender} | 💬 Message: {message_text}")

    # 3️⃣ Check if conversation exists
    conv = supabase.table("conversations").select("message_list").eq("session_id", session_id).eq("phone", int(phone)).single().execute()
    if not conv.data:
        print(f"🚫 No conversation found for {phone}")
        return {"success": True, "message": "No conversation found"}

    # 4️⃣ Store message
    store_message(session_id, int(phone), message_text, sender)

    # 5️⃣ Fetch and update conversation
    message_list = conv.data.get("message_list", [])
    message_list.append({sender: message_text})

    # 6️⃣ AI follow-up in background (non-blocking)
    async def handle_ai_followup():
        try:
            from llm_responses import generate_followup_message
            new_message = await generate_followup_message(message_list)

            print(f"🤖 AI generated follow-up: {new_message}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{WHATSAPP_SERVICE_URL}/session/{session_id}/send",
                    json={"to": phone_full, "text": new_message},
                    timeout=15.0
                )
                print(f"📤 Node.js send response: {response.status_code}")

            store_message(session_id, int(phone), new_message, "res_owner")

            await broadcast_to_websockets({
                "type": "new_message",
                "session_id": session_id,
                "message": {"text": new_message, "sender": "res_owner"}
            })
        except Exception as e:
            print(f"❌ Error generating/sending AI follow-up: {e}")

    background_tasks.add_task(handle_ai_followup)
    print(f"✅ Incoming message processed for {phone}")
    return {"success": True}



# ============= WEBSOCKET =============
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates"""
    print("🔌 New WebSocket connection request")
    await websocket.accept()
    print("✅ WebSocket connection accepted")
    websocket_connections.add(websocket)
    print(f"🌐 Total WebSocket connections: {len(websocket_connections)}")
    try:
        await websocket.send_json({"type": "connection_ack", "message": "WebSocket connected"})
        print("➡️ Entering WebSocket receive loop")
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

# ============= HEALTH CHECK =============
@router.get("/health")

async def health():
    print("🔍 Health check requested")
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "websocket_connections": len(websocket_connections),
    }