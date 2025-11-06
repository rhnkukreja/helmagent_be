import os
import uuid
from datetime import datetime
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from supabase import create_client, Client

# ============= CONFIG =============
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
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
    user_id: str

class SendMessageRequest(BaseModel):
    to: str
    text: str

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
def save_session_to_db(session_id: str, user_id: str, status: str, qr: str = None, phone: str = None):
    """Save or update session in Supabase"""
    try:
        existing = supabase.table("whatsapp_sessions").select("id").eq("id", session_id).execute()
        data = {
            "id": session_id,
            "user_id": user_id,
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
    session_id = str(uuid.uuid4())
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{WHATSAPP_SERVICE_URL}/session/start",
                json={"session_id": session_id},
                timeout=10.0
            )
            response.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start WA session: {str(e)}")

    active_sessions[session_id] = {
        "user_id": request.user_id,
        "status": "initializing",
        "qr": None,
        "phone": None,
        "created_at": datetime.utcnow()
    }
    save_session_to_db(session_id, request.user_id, "initializing")

    return {
        "session_id": session_id,
        "status": "initializing",
        "message": "Session started. Waiting for QR code..."
    }

@router.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get current session status and QR code"""
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
    except:
        pass
    raise HTTPException(status_code=404, detail="Session not found")

@router.post("/session/{session_id}/send")
async def send_message(session_id: str, request: SendMessageRequest):
    """Send a WhatsApp message"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if active_sessions[session_id]["status"] != "connected":
        raise HTTPException(status_code=400, detail="Session not connected")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{WHATSAPP_SERVICE_URL}/session/{session_id}/send",
                json={"to": request.to, "text": request.text},
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
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
    """Disconnect WhatsApp session"""
    async with httpx.AsyncClient() as client:
        try:
            await client.post(f"{WHATSAPP_SERVICE_URL}/session/{session_id}/disconnect", timeout=5.0)
        except:
            pass
    if session_id in active_sessions:
        del active_sessions[session_id]
    save_session_to_db(session_id, "", "disconnected")
    return {"success": True}

# ============= WEBHOOK =============
@router.post("/webhook")
async def whatsapp_webhook(webhook: WhatsAppWebhook):
    """Receive events from WhatsApp service"""
    session_id = webhook.session_id
    event = webhook.event
    data = webhook.data
    print(f"📥 Webhook received: {event} for session {session_id}")

    if session_id not in active_sessions:
        active_sessions[session_id] = {"user_id": "unknown", "status": "unknown", "qr": None, "phone": None}
    session = active_sessions[session_id]

    if event == "qr_ready":
        session.update({"status": "qr_ready", "qr": data.get("qr")})
        save_session_to_db(session_id, session["user_id"], "qr_ready", qr=data.get("qr"))
        await broadcast_to_websockets({"type": "qr_update", "session_id": session_id, "qr": data.get("qr")})

    elif event == "connected":
        session.update({"status": "connected", "phone": data.get("phone"), "qr": None})
        save_session_to_db(session_id, session["user_id"], "connected", phone=data.get("phone"))
        await broadcast_to_websockets({"type": "connected", "session_id": session_id, "phone": data.get("phone"), "name": data.get("name")})

    elif event == "disconnected":
        session["status"] = "disconnected"
        save_session_to_db(session_id, session["user_id"], "disconnected")
        await broadcast_to_websockets({"type": "disconnected", "session_id": session_id})

    elif event == "message_received":
        save_message_to_db(session_id, data)
        await broadcast_to_websockets({"type": "new_message", "session_id": session_id, "message": data})

    return {"success": True}

# ============= WEBSOCKET =============
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates"""
    await websocket.accept()
    websocket_connections.add(websocket)
    try:
        await websocket.send_json({"type": "connected", "message": "WebSocket connected"})
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

# ============= HEALTH CHECK =============
@router.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "websocket_connections": len(websocket_connections),
    }
