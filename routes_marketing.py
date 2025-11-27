from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from utils import safe_parse_items
import os
from dotenv import load_dotenv
from collections import defaultdict
from typing import List, Dict, Any, Optional
from llm_responses import generate_offer_message
from utils import format_phone_number, send_promo_message
from routes_whatsapp import send_to_whatsapp
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials not found in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
router = APIRouter(prefix="/marketing", tags=["marketing"])

# --- Models ---
# user_id is passed as a string/UUID, which is correct
class OfferCreate(BaseModel):
    user_id: str
    title: str
    description: str | None = None
    discount: str | None = None
    image_url: str | None = None

class GenerateMessagePayload(BaseModel):
    customer: Dict[str, Any]
    order_insights: Dict[str, Any]
    active_offer_ids: List[str]
    org_id: str

class CustomerData(BaseModel):
    name: str
    phone: str

class OfferUsed(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    discount: Optional[str] = None
    image: Optional[str] = None
    
class PromotionMessagePayload(BaseModel):
    org_id: str
    customer: CustomerData
    message: str
    offer_used: Optional[OfferUsed] = None
# --- Endpoints ---

@router.get("/offers/{user_id}")
def get_offers(user_id: str):
    """Fetch all draft and active offers for a specific user."""
    try:
        # Fetch offers ordered by creation date (newest first)
        response = supabase.table("offers")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()

        return {
            "success": True,
            # The structure for the frontend expects 'id', 'title', 'description', 'discount', 'image', 'created_at'
            # We map 'image_url' to 'image' for consistency with the frontend mock structure.
            "offers": [
                {
                    "id": offer["id"],
                    "title": offer["title"],
                    "description": offer["description"],
                    "discount": offer["discount"],
                    "created_at": offer["created_at"],
                    "image": offer["image_url"], # Mapping from image_url to image
                    "status": offer["status"], # Including status to filter for drafts/active
                    "sent_count": offer["sent_count"],  # Add this for active campaigns
                } for offer in response.data
            ]
        }
    except Exception as e:
        print(f"Error fetching offers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-offer")
def create_offer(payload: OfferCreate):
    """Create a new draft offer."""
    try:
        # Insert into Supabase
        data = supabase.table("offers").insert({
            "user_id": payload.user_id,
            "title": payload.title,
            "description": payload.description,
            "discount": payload.discount,
            "image_url": payload.image_url,
            # 'status' defaults to 'draft' as per your schema, no need to include
            # 'sent_count' defaults to 0, no need to include
        }).execute()

        # Check if data was returned
        if not data.data:
            raise HTTPException(status_code=500, detail="Failed to insert offer.")

        return {
            "success": True,
            "offer": data.data[0] # Return the created offer data
        }
    
    except Exception as e:
        print(f"Error creating offer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/activate-offer/{offer_id}")
def activate_offer(offer_id: str):
    """Activate a draft offer by setting its status to 'active'."""
    try:
        response = supabase.table("offers") \
            .update({"status": "active"}) \
            .eq("id", offer_id) \
            .execute()

        return {"success": True, "updated": response.data}
    except Exception as e:
        print(f"Error activating offer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW ENDPOINT: DEACTIVATE Offer ---
@router.post("/deactivate-offer/{offer_id}")
def deactivate_offer(offer_id: str):
    """Deactivate an active offer by setting its status to 'inactive'."""
    try:
        response = supabase.table("offers") \
            .update({"status": "inactive"}) \
            .eq("id", offer_id) \
            .execute()

        return {"success": True, "updated": response.data}
    except Exception as e:
        print(f"Error deactivating offer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- DELETE Offer ---
@router.delete("/offers/{offer_id}")
async def delete_offer(offer_id: str):
    try:
        res = supabase.table("offers").delete().eq("id", offer_id).execute()
        
        # Correct way to check if delete worked
        if res.count == 0:  # or len(res.data) == 0 and res.count is None
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "Offer not found"}
            )
            
        return {"success": True, "message": "Offer deleted"}
        
    except Exception as e:
        print("Delete error:", e)
        raise HTTPException(status_code=500, detail="Failed to delete offer")

@router.post("/increment-sent/{offer_id}")
def increment_sent(offer_id: str):
    try:
        supabase.rpc("increment_offer_sent", {"offer_id": offer_id}).execute()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/customers/{user_id}")
def get_customer_audience(user_id: str):
    """Fetch aggregated customer data for the audience list by calling the Supabase RPC."""
    if not supabase: raise HTTPException(status_code=503, detail="Database not connected.")
    try:
        # user_id is passed as org_id (p_org_id) to the RPC
        response = supabase.rpc("get_customer_audience_data", {"p_org_id": user_id}).execute()
        
        # Format the total_order_cost as a currency string (e.g., "₹ 12,450.00")
        formatted_data = [
            {
                # Use the normalized contact number as the primary ID for the customer record
                "id": data["normalized_contact"], 
                "name": data["canonical_name"],
                "phone": data["normalized_contact"], # Use the normalized contact for display/lookup
                "totalOrders": data["total_orders"],
                "lastVisit": data["last_visit"], 
                "segment": "Standard", 
                # Format with commas and 2 decimal places, prepended by the currency symbol
                "totalOrderCost": f"₹ {data['total_order_cost']:,.2f}", 
                "latestItem": data["latest_item"] or "N/A", 
            } for data in response.data
        ]

        return {
            "success": True,
            "customers": formatted_data
        }

    except Exception as e:
        print(f"Error fetching customer audience data: {e}")
        # Return a 500 with the specific database error detail
        raise HTTPException(status_code=500, detail=f"Failed to fetch customer data: {e}")

@router.get("/customer/order-insights")
def get_customer_order_insights(phone: str, org_id: str):
    try:
        # Normalize phone
        digits_only = ''.join(filter(str.isdigit, phone))
        base_number = digits_only[-10:]

        possible_numbers = [
            base_number,
            f"91{base_number}",
            f"+91{base_number}",
            phone
        ]

        # Fetch all bills for this customer
        res = (
            supabase.table("bills")
            .select("*")
            .eq("org_id", org_id)
            .in_("contact_number", possible_numbers)
            .order("bill_date", desc=True)
            .execute()
        )

        if not res.data:
            return {
                "success": False,
                "message": "No orders found for this customer"
            }

        bills = res.data
        latest_bill = bills[0]

        # Parse latest bill (full order)
        full_latest_order = safe_parse_items(latest_bill.get("items_ordered"))

        # -------------------------------
        # AGGREGATION LOGIC
        # -------------------------------
        item_qty = defaultdict(int)
        item_revenue = defaultdict(float)
        item_label_map = {}  # use this to return proper item names

        for bill in bills:
            items = safe_parse_items(bill.get("items_ordered"))
            if not items:
                continue

            for item in items:
                raw_name = (
                    item.get("item_name")
                    or item.get("name")
                    or item.get("title")
                    or "Unknown Item"
                ).strip()

                normalized_key = raw_name.casefold()
                item_label_map[normalized_key] = raw_name

                # Parse quantity safely
                qty_raw = item.get("quantity", 1)
                try:
                    qty = int(float(str(qty_raw)))
                except:
                    qty = 1

                # Parse price safely
                price_raw = item.get("price") or item.get("rate") or 0
                try:
                    price = float(str(price_raw).replace(",", ""))
                except:
                    price = 0.0

                # Accumulate
                item_qty[normalized_key] += qty
                item_revenue[normalized_key] += qty * price

        # -------------------------------
        # FIND MOST ORDERED ITEM (BY TOTAL REVENUE)
        # -------------------------------
        if item_revenue:
            best_key = max(item_revenue.items(), key=lambda x: x[1])[0]
            most_ordered_item = item_label_map[best_key]
            most_ordered_qty = item_qty[best_key]
            most_ordered_revenue = round(item_revenue[best_key], 2)
        else:
            most_ordered_item = "No items ordered"
            most_ordered_qty = 0
            most_ordered_revenue = 0.0

        # -------------------------------
        # RETURN RESULT
        # -------------------------------
        return {
            "success": True,
            "name": latest_bill.get("name") or "Unknown",
            "phone": phone,
            "last_visit_date": latest_bill.get("bill_date"),

            # Latest bill full items
            "full_latest_order": full_latest_order,

            # MOST ORDERED ITEM — FINAL RESULT
            "most_ordered_item": most_ordered_item,
            "most_ordered_quantity": most_ordered_qty,
            "most_ordered_revenue": most_ordered_revenue,

            # Total bills counted
            "total_orders_found": len(bills)
        }

    except Exception as e:
        print("Error in order-insights:", e)
        return {"success": False, "error": str(e)}

@router.post("/generate-message")
async def generate_message(payload: GenerateMessagePayload):
    try:
        print("\n" + "*" * 120)

        org_id = payload.org_id
        customer = payload.customer
        insights = payload.order_insights or {}
        offer_ids = payload.active_offer_ids or []

        # 1. FETCH ACTIVE OFFERS
        if offer_ids:
            offers_response = (
                supabase.table("offers")
                .select("id, title, discount, description, image_url")
                .eq("user_id", org_id)
                .in_("id", offer_ids)
                .eq("status", "active")
                .execute()
            )
            active_offers = offers_response.data or []
        else:
            active_offers = []

        if not active_offers:
            return {
                "success": False,
                "message": "No active offers available to generate a message."
            }

        # 2. DELEGATE TO LLM SERVICE
        final_message, best_offers = await generate_offer_message(
            customer=customer,
            insights=insights,
            active_offers=active_offers,
        )

        return {
            "success": True,
            "message": final_message,
            "offers_used": best_offers,
        }

    except Exception as e:
        print("Generate message error:", e)
        return {
            "success": False,
            "message": "Message generation failed."
        }

@router.post("/send-promotion-message")
async def send_promotion_message(payload: PromotionMessagePayload):

    print("\n-------- NEW PROMOTION MESSAGE --------")
    print(f"ORG ID        : {payload.org_id}")
    print(f"CUSTOMER NAME : {payload.customer.name}")
    print(f"CUSTOMER PHONE: {payload.customer.phone}")
    print(f"MESSAGE       : {payload.message}")
    image_url = None
    if payload.offer_used:
        print("OFFER USED:")
        print(f"  ID        : {payload.offer_used.id}")
        print(f"  Title     : {payload.offer_used.title}")
        print(f"  Discount  : {payload.offer_used.discount}")
        print(f"  Image URL : {payload.offer_used.image}")
        image_url = payload.offer_used.image
    else:
        print("OFFER USED   : None")
    print("---------------------------------------\n")
    org_id = payload.org_id
    session_id = payload.org_id
    phone = payload.customer.phone
    text = payload.message

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
    if not image_url:
        response_data = await send_to_whatsapp(session_id, formatted_phone, text)
    else:
        print("Image found so havent sent the message")
        response_data = await send_promo_message(session_id, formatted_phone, text, image_url)
    # You can    later replace print(...) with sending message via WhatsApp / SMS
    return {"success": True, "detail": "Message received successfully"}