from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import razorpay
import os
from dotenv import load_dotenv

# ============= CONFIG =============
load_dotenv()
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    raise RuntimeError("Razorpay live key ID or secret missing in .env")

client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# In-memory store for temporary tracking (optional)
payment_status_store = {}

router = APIRouter(prefix="/razorpay", tags=["RazorPay"])


@router.post("/create-order")
async def create_order(payload: dict):
    """
    Create a Razorpay order.
    Expected payload: { "amount": 100, "currency": "INR", "receipt": "order_rcptid_11" }
    """
    try:
        print("here")
        print("Creating Razorpay order with payload:", payload)
        amount = int(payload.get("amount", 0))
        if amount <= 0:
            raise ValueError("Invalid amount")

        order = client.order.create({
            "amount": amount * 100,
            "currency": payload.get("currency", "INR"),
            "receipt": payload.get("receipt", f"rcpt_{os.urandom(4).hex()}"),
            "payment_capture": 1
        })
        print("Razorpay order created:", order)
        payment_status_store[order["id"]] = {"status": "created", "order": order}
        return JSONResponse({
            "order_id": order["id"],
            "amount": order["amount"],
            "currency": order["currency"],
            "receipt": order["receipt"],
            "status": order["status"]
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------- Verify Payment ----------------------
@router.post("/verify-payment")
async def verify_payment(payload: dict):
    """
    Verify payment using Razorpay signature.
    Expected payload:
    {
      "razorpay_order_id": "...",
      "razorpay_payment_id": "...",
      "razorpay_signature": "..."
    }
    """
    try:
        client.utility.verify_payment_signature(payload)
        payment_id = payload.get("razorpay_payment_id")
        payment = client.payment.fetch(payment_id)
        payment_status_store[payment["order_id"]] = {
            "status": payment["status"],
            "payment": payment
        }
        return JSONResponse({"status": "success", "payment": payment})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signature verification failed: {e}")


# ---------------------- Get Payment Status ----------------------
@router.get("/payment-status/{payment_id}")
async def payment_status(payment_id: str):
    """
    Retrieve the payment status from Razorpay or local cache.
    """
    try:
        if payment_id in payment_status_store:
            return {"source": "local", "data": payment_status_store[payment_id]}

        payment = client.payment.fetch(payment_id)
        return {"source": "razorpay", "data": payment}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Payment not found: {e}")


# ---------------------- Create Payment Link ----------------------
@router.post("/create-payment-link")
async def create_payment_link(payload: dict):
    """
    Create a shareable payment link for your customer.
    Expected payload: { "amount": 100, "customer_name": "John", "email": "...", "contact": "9999999999" }
    """
    try:
        amount = int(payload.get("amount", 0))
        if amount <= 0:
            raise ValueError("Invalid amount")

        link = client.payment_link.create({
            "amount": amount * 100,
            "currency": "INR",
            "accept_partial": False,
            "description": "Hospitality Service Payment",

            "callback_url": "https://8073703458c6.ngrok-free.app/payment-success",
            "callback_method": "get",

            "customer": {
                "name": payload.get("customer_name", "Guest"),
                "email": payload.get("email"),
                "contact": payload.get("contact")
            },
            "notify": {"sms": True, "email": True},
            "reminder_enable": True 
        })

        print("Payment link created:", link)
        return {"link_id": link["id"], "short_url": link["short_url"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------- Create UPI QR Code ----------------------
@router.post("/create-qr-code")
async def create_qr_code(payload: dict):
    """
    Create a single-use UPI QR code for on-premise payments.
    Expected payload: { "amount": 500, "description": "Hotel booking" }
    """
    try:
        amount = int(payload.get("amount", 0))
        qr = client.qr_code.create({
            "type": "upi_qr",
            "name": "Hospitality Payment",
            "usage": "single_use",
            "fixed_amount": True,
            "payment_amount": amount * 100,
            "description": payload.get("description", "Room Payment")
        })
        return {"qr_id": qr["id"], "image_url": qr["image_url"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/create-subscription")
async def create_subscription(payload: dict):
    """
    Create Razorpay subscription for auto-debit after 14 days.
    Expected payload: { "plan_id": "...", "customer_name": "...", "email": "...", "contact": "..." }
    """
    print("Creating subscription with payload:", payload)
    import time
    
    try:
        subscription = client.subscription.create({
            "plan_id": payload["plan_id"],   # Example: plan_EqX976dnj3bFGv
            "total_count": 12,               # 12 months
            "quantity": 1,
            "customer_notify": 1,
            "start_at": int(time.time()) + (14 * 24 * 60 * 60),  # ⏳ starts after 14 days
            "addons": [],
            "notes": {
                "customer_name": payload["customer_name"],
            }
        })
        print("\n" + "="*50)
        print("RAZORPAY SUBSCRIPTION CREATED")
        print("="*50)
        print(f"Subscription ID : {subscription['id']}")
        print(f"Short URL       : {subscription['short_url']}")
        print(f"Status          : {subscription['status']}")
        print(f"Plan ID         : {subscription['plan_id']}")
        print(f"Start At        : {subscription['start_at']} (14 days from now)")
        print(f"Customer Name   : {payload['customer_name']}")
        print(f"Email           : {payload.get('email', 'N/A')}")
        print(f"Contact (org_id): {payload.get('contact', 'N/A')}")
        print("="*50 + "\n")
        # This link will open Razorpay Checkout + save card token
        return {
            "subscription_id": subscription["id"],
            "short_url": subscription["short_url"]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------- Refund Payment ----------------------
@router.post("/refund")
async def refund_payment(payload: dict):
    """
    Initiate refund for a completed payment.
    Expected payload: { "payment_id": "...", "amount": 100 }
    """
    try:
        refund = client.payment.refund(
            payload["payment_id"],
            {"amount": payload.get("amount", 0) * 100}
        )
        return {"refund_id": refund["id"], "status": refund["status"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
