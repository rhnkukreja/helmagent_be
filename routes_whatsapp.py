import os
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Set, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
import httpx
import requests
import traceback
from collections import defaultdict
from dotenv import load_dotenv

from supabase import create_client, Client
from llm_responses import generate_followup_message
from utils import update_contact_status, format_phone_number, fetch_rest_detail

# ============= LOGGING =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# ============= HELPERS =============
def normalize_phone(phone: str) -> str:
    """Normalize phone to: 919999000001 (no +, no spaces)"""
    if not phone:
        return ""
    digits = ''.join(c for c in str(phone) if c.isdigit())
    # If 10 digits, assume India
    if len(digits) == 10:
        digits = '91' + digits
    return digits

async def broadcast_to_org(org_id: str, message: dict):
    """Send message ONLY to WebSocket clients of a specific org_id"""
    if org_id not in websocket_connections:
        return

    dead_ws = set()
    for ws in websocket_connections[org_id].copy():
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.warning(f"WebSocket send failed: {e}")
            dead_ws.add(ws)

    for ws in dead_ws:
        websocket_connections[org_id].discard(ws)

    logger.info(f"Broadcasted to org_id {org_id}: {message.get('type')}")

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
            logger.info(f"Updated session {session_id} → {status}")
        else:
            data["created_at"] = datetime.utcnow().isoformat()
            supabase.table("whatsapp_sessions").insert(data).execute()
            logger.info(f"Created session {session_id} → {status}")
    except Exception as e:
        logger.error(f"DB Error saving session: {e}")

def store_message_sync(session_id: str, phone: str, message: str, sender: str, org_id: str = None):
    """Synchronous message storage (run in thread pool)"""
    try:
        logger.info(f"Storing message for {phone} (Session: {session_id})")

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
                logger.info(f"Fetching customer details for org_id={org_id}")
                customer_details = fetch_customer_details(org_id, phone)
                if customer_details:
                    upsert_data.update({
                        "name": customer_details.get("name"),
                        "bill_amount": customer_details.get("bill_amount"),
                        "bill_date": customer_details.get("bill_date"),
                    })
                    logger.info(f"Added customer details: {customer_details}")

        res = (
            supabase.table("conversations")
            .upsert(upsert_data, on_conflict="session_id, phone")
            .execute()
        )

        logger.info(f"Message stored successfully for {phone}")
        return res.data

    except Exception as e:
        logger.error(f"Error storing message: {e}")
        return None

async def store_message_async(session_id: str, phone: str, message: str, sender: str, org_id: str = None):
    """Async wrapper for store_message_sync"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, store_message_sync, session_id, phone, message, sender, org_id)

def fetch_customer_details(org_id: str, phone: str):
    """Fetch customer details from bills table"""
    try:
        
        logger.info(f"Fetching customer details for phone: {phone}, org_id: {org_id}")
        
        res = (
            supabase.table("bills")
            .select("name, bill_date, total_amount")
            .eq("org_id", org_id)
            .eq("contact_number", phone)
            .order("bill_date", desc=True)
            .limit(1)
            .execute()
        )

        if res.data and len(res.data) > 0:
            customer = res.data[0]
            name = customer.get("name")
            bill_amount = customer.get("total_amount")
            bill_date = customer.get("bill_date")

            logger.info(f"Found customer: {name}, Bill: {bill_amount}, Date: {bill_date}")
            return {
                "name": name,
                "bill_date": bill_date,
                "bill_amount": str(bill_amount) if bill_amount else None,
            }

        logger.warning(f"No bill found for {phone} in org {org_id}")
        return None

    except Exception as e:
        logger.error(f"Error fetching customer details: {e}")
        return None

async def send_to_whatsapp(session_id: str, phone: str, text: str, max_retries: int = 3) -> dict:
    """
    Reusable send logic with retries.
    Returns response dict or raises HTTPException
    """
    if phone.startswith('+91') or phone.startswith('+1'):   # change needed here to check for 91 numbers without the +
        # Remove the '+' sign as requested
        formatted_phone = phone[1:]
        print("➡️ Phone number already has country code, formatted as:", formatted_phone)
        
    elif phone.startswith('91') or phone.startswith('1'):
        # Already has country code without '+'
        formatted_phone = phone
        print("➡️ Phone number already has country code, formatted as:", formatted_phone)
    else:
        try:
            # Add country code from org_id (assuming format_phone_number adds it)
            formatted_phone = format_phone_number(org_id, phone)
        except Exception as format_error:
            raise HTTPException(status_code=400, detail=f"Invalid phone format: {str(format_error)}")

    last_exception = None

    async with httpx.AsyncClient() as client:
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Send attempt {attempt}/{max_retries} to {formatted_phone}")

                response = await client.post(
                    f"{WHATSAPP_SERVICE_URL}/session/{session_id}/send",
                    json={"to": phone, "text": text},
                    timeout=30.0
                )

                # 4xx errors = don't retry
                if 400 <= response.status_code < 500:
                    if "Session not connected" in response.text:
                        logger.warning("Session not connected. Retrying...")
                        await asyncio.sleep(2)
                        continue
                    logger.error(f"Client error {response.status_code}: {response.text}")
                    raise HTTPException(status_code=response.status_code, detail=response.text)

                response.raise_for_status()
                logger.info(f"✅ Message sent successfully on attempt {attempt}")
                return response.json()

            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
                last_exception = e
                logger.warning(f"Attempt {attempt} failed: {str(e)}")

                if attempt < max_retries:
                    logger.info(f"Waiting 2s before retry {attempt + 1}...")
                    await asyncio.sleep(2)

    logger.error(f"All {max_retries} attempts exhausted")
    raise HTTPException(status_code=504, detail=f"Failed after {max_retries} attempts: {str(last_exception)}")

def format_whatsapp_message(llm_response: str) -> str:
    """Format LLM response for WhatsApp readability"""
    import re

    sentences = re.split(r'(?<=[.!?])\s+', llm_response)
    formatted_lines = []
    buffer = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(buffer) + len(sentence) > 100 and buffer:
            formatted_lines.append(buffer)
            buffer = sentence
        else:
            buffer += " " + sentence if buffer else sentence

    if buffer:
        formatted_lines.append(buffer)

    return "\n\n".join(formatted_lines)

# ============= ENDPOINTS =============

@router.post("/session/start")
async def start_session(request: StartSessionRequest):
    """Start a new WhatsApp session and generate QR"""
    session_id = request.org_id
    logger.info(f"Starting WhatsApp session for org_id: {session_id}")

    try:
        async with httpx.AsyncClient() as client:
            url = f"{WHATSAPP_SERVICE_URL}/session/start"
            logger.info(f"Requesting: {url}")
            response = await client.post(
                url,
                json={"session_id": session_id},
                timeout=10.0
            )
            response.raise_for_status()
            logger.info(f"Node.js response: {response.status_code}")
    except Exception as e:
        logger.error(f"Error starting WA session: {e}")
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
    """Get session status"""
    logger.info(f"Fetching status for session: {session_id}")

    if session_id in active_sessions:
        return active_sessions[session_id]

    try:
        result = supabase.table("whatsapp_sessions").select("*").eq("id", session_id).single().execute()

        if result.data:
            return {
                "session_id": session_id,
                "status": result.data.get("status"),
                "qr": result.data.get("qr_code"),
                "phone": result.data.get("phone_number")
            }

    except Exception as e:
        logger.error(f"Error fetching session status: {e}")

    raise HTTPException(status_code=404, detail="Session not found")

@router.post("/send-whatsapp")
async def send_whatsapp(request: Request):
    """Send WhatsApp message with retries"""
    try:
        body = await request.json()
        logger.info(f"Send WhatsApp request: {body}")

        org_id = body.get("org_id")
        phone = body.get("phone")
        text = body.get("message")
        logger.info(f"Preparing to send message to {phone} for org_id {org_id}")
        if not org_id or not phone or not text:
            raise HTTPException(status_code=400, detail="org_id, phone, and message are required")

        # Check session exists and is connected
        res = supabase.table("whatsapp_sessions").select("id, status").eq("id", org_id).execute()

        if not res.data:
            raise HTTPException(status_code=404, detail="No WhatsApp session found for this org_id")

        session_info = res.data[0]
        session_id = session_info.get("id")
        status = session_info.get("status")

        if status != "connected":
            logger.warning(f"Session {session_id} not connected: {status}")
            raise HTTPException(status_code=400, detail=f"WhatsApp session is {status}, not connected")

        # Send message (with retries)
        if phone.startswith('+91') or phone.startswith('+1'):   # change needed here to check for 91 numbers without the +
            # Remove the '+' sign as requested
            formatted_phone = phone[1:]
            print("➡️ Phone number already has country code, formatted as:", formatted_phone)
            
        elif phone.startswith('91') or phone.startswith('1'):
            # Already has country code without '+'
            formatted_phone = phone
            print("➡️ Phone number already has country code, formatted as:", formatted_phone)
        else:
            try:
                # Add country code from org_id (assuming format_phone_number adds it)
                formatted_phone = format_phone_number(org_id, phone)
            except Exception as format_error:
                raise HTTPException(status_code=400, detail=f"Invalid phone format: {str(format_error)}")

        response_data = await send_to_whatsapp(session_id, formatted_phone, text)

        # Store message
        logger.info(f"Storing message for +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++__________________- {phone}")
        
        await store_message_async(session_id, phone, text, "res_owner", org_id)
        logger.info(f"Message stored for {formatted_phone}")
        update_contact_status(org_id, phone)

        logger.info(f"Message sent to {formatted_phone} successfully")
        return {"success": True, "data": response_data}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in send_whatsapp: {e}", exc_info=True)
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
        logger.error(f"Error fetching messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/{session_id}/disconnect")
async def disconnect_whatsapp(session_id: str):
    """Disconnect WhatsApp session"""
    logger.info(f"Disconnecting session: {session_id}")

    try:
        async with httpx.AsyncClient() as client:
            node_res = await client.post(
                f"{WHATSAPP_SERVICE_URL}/session/{session_id}/disconnect",
                timeout=10.0
            )
            logger.info(f"Node disconnect: {node_res.status_code}")
    except Exception as err:
        logger.warning(f"Node disconnect failed (not fatal): {err}")

    try:
        supabase.table("whatsapp_sessions").update({
            "status": "disconnected",
            "auth_data": None,
            "qr_code": None,
            "phone_number": None,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", session_id).execute()

        logger.info(f"Session {session_id} marked disconnected in DB")

    except Exception as e:
        logger.error(f"Error updating DB: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "success": True,
        "message": "WhatsApp session disconnected successfully",
        "session_id": session_id
    }

# ============= WEBHOOKS =============

@router.post("/webhook/whatsapp")
async def whatsapp_webhook_handler(webhook: WhatsAppWebhook):
    """Route webhook events"""
    logger.info(f"Webhook event: {webhook.event}")

    if webhook.event == "qr_ready":
        return await whatsapp_qr(webhook)
    elif webhook.event == "connected":
        return await whatsapp_connected(webhook)
    elif webhook.event == "disconnected":
        return await whatsapp_disconnected(webhook)
    elif webhook.event == "message_received":
        return await whatsapp_message(webhook, BackgroundTasks())
    elif webhook.event == "user_logout":
        return await whatsapp_user_logout(webhook)

    logger.warning(f"Unknown event: {webhook.event}")
    return {"success": False, "error": "Unknown event"}

@router.post("/webhook/qr")
async def whatsapp_qr(webhook: WhatsAppWebhook):
    """Handle QR code generation"""
    session_id = webhook.session_id
    data = webhook.data
    logger.info(f"QR ready for session {session_id}")

    if session_id not in active_sessions:
        active_sessions[session_id] = {"status": "qr_ready"}
    else:
        active_sessions[session_id]["status"] = "qr_ready"

    expires_in = data.get("expires_in", 60)
    expires_at = (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat()

    try:
        supabase.table("whatsapp_sessions").update({
            "status": "qr_ready",
            "qr_code": data.get("qr"),
            "qr_expires_at": expires_at,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", session_id).execute()
    except Exception as e:
        logger.error(f"Error saving QR: {e}")

    await broadcast_to_org(session_id, {
        "type": "qr_update",
        "session_id": session_id,
        "qr": data.get("qr"),
        "expires_in": expires_in
    })

    return {"success": True}

@router.post("/webhook/connected")
async def whatsapp_connected(webhook: WhatsAppWebhook):
    """Handle WhatsApp connection"""
    session_id = webhook.session_id
    data = webhook.data
    phone = data.get("phone")

    logger.info(f"Connected: {session_id} → {phone}")

    # Get org_id from DB (saved during /session/start)
    try:
        result = supabase.table("whatsapp_sessions").select("user_id").eq("id", session_id).execute()
        org_id = result.data[0]["user_id"] if result.data else session_id
    except Exception as e:
        logger.warning(f"Could not fetch org_id from DB: {e}, using session_id")
        org_id = session_id

    active_sessions[session_id] = {
        "status": "connected",
        "phone": phone,
        "org_id": org_id
    }

    save_session_to_db(session_id, org_id, "connected", phone=phone)

    await broadcast_to_org(session_id, {
        "type": "connected",
        "session_id": session_id,
        "phone": phone
    })

    return {"success": True}

@router.post("/webhook/disconnect")
async def whatsapp_disconnected(webhook: WhatsAppWebhook):
    """Handle disconnection"""
    session_id = webhook.session_id
    reason = webhook.data.get("reason", "unknown")

    logger.info(f"Disconnected: {session_id} — {reason}")

    if session_id in active_sessions:
        del active_sessions[session_id]

    save_session_to_db(session_id, session_id, "disconnected")

    await broadcast_to_org(session_id, {
        "type": "disconnected",
        "session_id": session_id,
        "reason": reason
    })

    return {"success": True}

@router.post("/webhook/message")
async def whatsapp_message(webhook: WhatsAppWebhook, background_tasks: BackgroundTasks):
    """Handle incoming messages"""
    session_id = webhook.session_id
    data = webhook.data

    logger.info(f"Message received for session: {session_id}")

    remote_jid = data.get("from", "")
    phone = None

    # Extract phone from JID
    if "@s.whatsapp.net" in remote_jid:
        phone = remote_jid.replace("@s.whatsapp.net", "").replace("+", "").strip()
    elif "@lid" in remote_jid:
        raw_data = data.get("raw", {})
        key_data = raw_data.get("key", {})
        sender_pn = key_data.get("senderPn", "")
        if sender_pn and "@s.whatsapp.net" in sender_pn:
            phone = sender_pn.replace("@s.whatsapp.net", "").replace("+", "").strip()

    if not phone:
        logger.warning(f"Could not extract phone from JID: {remote_jid}")
        return {"success": True}

    # Extract message
    message_obj = data.get("message", {})
    if isinstance(message_obj, dict):
        message_text = message_obj.get("text", "")
    elif isinstance(message_obj, str):
        message_text = message_obj
    else:
        message_text = ""

    is_from_me = data.get("fromMe", False)
    sender = "res_owner" if is_from_me else "res_customer"

    logger.info(f"From: {sender} | Message: {message_text[:50]}...")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++=++++++++++++++++++++++++++++++")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++=++++++++++++++++++++++++++++++")
    print("phone:", phone)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++=++++++++++++++++++++++++++++++")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++=++++++++++++++++++++++++++++++")
    # Check if conversation exists
    try:
        conv = supabase.table("conversations") \
            .select("message_list") \
            .eq("session_id", session_id) \
            .eq("phone", phone) \
            .execute()
    except Exception as e:
        logger.error(f"DB error: {e}")
        return {"success": False, "error": "DB error"}

    if not conv.data:
        logger.info(f"No conversation for {phone} — ignoring")
        return {"success": True}

    # Store message and trigger AI
    await store_message_async(session_id, phone, message_text, sender)

    async def handle_ai_followup():
        try:
            logger.info("Starting AI followup task")

            result = supabase.table("conversations") \
                .select("message_list") \
                .eq("session_id", session_id) \
                .eq("phone", phone) \
                .execute()

            if not result.data:
                logger.warning(f"Conversation not found for AI reply")
                return

            message_list = result.data[0].get("message_list", [])
            print("_________________________________here")
            restaurant_name, google_review_link = fetch_rest_detail(session_id)
            print("_______________________________________________________rest")
            logger.info(f"Restaurant: {restaurant_name}")
            logger.info(f"google_review_link: {google_review_link}")
            print("google_review_link:", google_review_link)
            ai_message = await generate_followup_message(message_list, restaurant_name, google_review_link)
            formatted = format_whatsapp_message(ai_message)

            logger.info(f"Sending AI reply: {formatted[:50]}...")
            
            await send_to_whatsapp(session_id, phone, formatted)
            await store_message_async(session_id, phone, formatted, "res_owner")

            await broadcast_to_org(session_id, {
                "type": "new_message",
                "session_id": session_id,
                "message": {"text": formatted, "sender": "res_owner"}
            })

            logger.info("AI followup completed")

        except Exception as e:
            logger.error(f"AI followup error: {e}", exc_info=True)

    background_tasks.add_task(handle_ai_followup)
    return {"success": True}

@router.post("/webhook/user-logout")
async def whatsapp_user_logout(webhook: WhatsAppWebhook):
    """Handle logout"""
    session_id = webhook.session_id
    data = webhook.data
    reason = data.get("reason", "unknown")

    logger.info(f"User logout: {session_id} — {reason}")

    try:
        supabase.table("whatsapp_sessions").update({
            "user_id": "",
            "status": "logged_out",
            "auth_data": None,
            "phone_number": None
        }).eq("id", session_id).execute()

        await broadcast_to_org(session_id, {
            "type": "user_logout",
            "session_id": session_id,
            "status": "logged_out",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        })

        logger.info(f"Logout processed for {session_id}")
        return {"received": True, "message": "Logout processed"}

    except Exception as e:
        logger.error(f"Error processing logout: {e}")
        return {"received": False, "error": str(e)}

# ============= WEBSOCKET =============

@router.websocket("/ws/{org_id}")
async def websocket_endpoint(websocket: WebSocket, org_id: str):
    """WebSocket connection for real-time updates"""
    logger.info(f"WebSocket connection request: {org_id}")
    await websocket.accept()
    logger.info(f"WebSocket accepted: {org_id}")

    websocket_connections[org_id].add(websocket)

    try:
        await websocket.send_json({
            "type": "connection_ack",
            "message": "WebSocket connected",
            "org_id": org_id
        })

        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {org_id}")
        websocket_connections[org_id].discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket error {org_id}: {e}")
        websocket_connections[org_id].discard(websocket)

# ============= HEALTH CHECK =============

@router.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "websocket_connections": sum(len(conns) for conns in websocket_connections.values()),
    }



if __name__=="__main__":
    org_id="40cd4216-c63e-434f-89f3-600cbaa5d93e"
    phone="+918894615869"
    (fetch_customer_details(org_id, phone))
    
