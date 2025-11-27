import os 
import json
from supabase import create_client, Client
from dotenv import load_dotenv
import httpx
import base64
import logging
import asyncio
from fastapi import HTTPException
load_dotenv()

SUPABASE_KEY=os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_URL=os.getenv("SUPABASE_URL")
WHATSAPP_SERVICE_URL=os.getenv("WHATSAPP_SERVICE_URL")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

supabase: Client = create_client(SUPABASE_URL,SUPABASE_KEY)
def store_in_supabase(extracted_data: dict):
    """
    Stores extracted data into the Supabase table, now with bill_date and total_amount.
    """
    try:
        print("Storing extracted data in Supabase...")
        record = {
            "name": extracted_data.get("name", ""),
            "contact_number": extracted_data.get("contact_number", ""),
            "items_ordered": extracted_data.get("items_ordered", []),
            # expect date as 'YYYY-MM-DD' or empty string
            "bill_date": extracted_data.get("date") or None,
            # store numeric total - if missing, store None/0.0 depending on your preference
            "total_amount": extracted_data.get("total_amount", ""), 
            "order_type": extracted_data.get("order_type", ""),
            # keep org_id if present
            "org_id": extracted_data.get("org_id", None),
        }
        print("Record to insert:", record)
        response = supabase.table("bills").insert(record).execute()

        if response.data:
            return response.data
        else:
            raise Exception(str(response))

    except Exception as e:
        print("Error storing in Supabase:", str(e))
        raise HTTPException(status_code=500, detail=f"Supabase error: {str(e)}")

def update_contact_status(org_id: str, phone: str):
    print(f"📌 Updating contact status for phone={phone}, org_id={org_id}")

    try:
        # Try to update the record
        result = supabase.table("bills") \
            .update({"status": True}) \
            .eq("contact_number", phone) \
            .eq("org_id", org_id) \
            .execute()

        print("✅ Contact status updated:", result.data)

    except Exception as e:
        print(f"❌ Failed to update contact status: {str(e)}")



def format_phone_number(org_id: str, phone: str):
    """
    Fetch country code using org_id and return properly formatted phone number.
    Removes '+' and spaces and builds: countrycode + phonenumber
    """

    print(f"➡️ Formatting phone number for org_id: {org_id}, phone: {phone}")

    # Fetch country_code from organizations table
    org_data = supabase.table("organizations") \
        .select("country_code") \
        .eq("org_id", org_id) \
        .execute()

    print("➡️ Fetched organization data:", org_data.data)

    # --- FIX 1: handle empty result safely ---
    if not org_data.data or len(org_data.data) == 0:
        print("⚠️ No organization found. Using default country code +91")
        country_code = "91"
    else:
        # --- FIX 2: org_data.data[0] is the row ---
        raw_code = org_data.data[0].get("country_code") or "+91"
        print(f"➡️ Raw country code from DB: {raw_code}")

        # Clean the country code
        country_code = raw_code.strip().lstrip("+")

    print(f"➡️ Cleaned country code: {country_code}")

    # Clean incoming phone number
    clean_phone = str(phone).strip().replace(" ", "").lstrip("+")
    print(f"➡️ Cleaned phone: {clean_phone}")

    # Build final WhatsApp-ready number
    formatted = f"{country_code}{clean_phone}"
    print(f"📱 Final formatted phone: {formatted}")

    return formatted

def fetch_rest_detail(org_id: str):
    """
    Fetch restaurant details like name and WhatsApp number using org_id.
    """
    print(f"➡️ Fetching restaurant details for org_id: {org_id}")

    try:
        rest_data = supabase.table("organizations") \
            .select("name, google_review_link") \
            .eq("org_id", org_id) \
            .execute()

        print("➡️ Fetched restaurant data:", rest_data.data)
        if rest_data.data and len(rest_data.data) > 0:
            name = rest_data.data[0].get("name", "Unknown")
            link = rest_data.data[0].get("google_review_link", "")
            return name,link
        else:
            raise Exception("No organization found with the given org_id.")

    except Exception as e:
        print(f"❌ Error fetching restaurant details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Supabase error: {str(e)}")


def safe_parse_items(items_raw):
    """
    Accepts ANY form of items_ordered:
    - JSONB array (list)
    - Raw JSON string
    - Escaped JSON string
    - Bad string
    - None
    Returns a LIST always.
    """
    # Case 1: Already JSONB (list)
    if isinstance(items_raw, list):
        return items_raw

    # Case 2: None or empty string
    if not items_raw:
        return []

    # Case 3: Stored as STRING (escaped or normal)
    if isinstance(items_raw, str):
        try:
            # First try direct json.loads
            return json.loads(items_raw)
        except:
            try:
                # Try unescaping then parsing
                unescaped = items_raw.encode('utf-8').decode('unicode_escape')
                return json.loads(unescaped)
            except:
                return []

    # Default fallback
    return []

# converting image to base 64 
async def url_to_base64(url: str) -> str:
    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        res.raise_for_status()

    encoded = base64.b64encode(res.content).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


async def send_promo_message(session_id: str, phone: str, text: str, image_url: str, max_retries: int = 3):

# Sends an IMAGE + CAPTION WhatsApp message with retry logic.

    last_exception = None
    base64_img = await url_to_base64(image_url)

    async with httpx.AsyncClient() as client:
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[IMAGE] Attempt {attempt}/{max_retries} → {phone}")

                response = await client.post(
                    f"{WHATSAPP_SERVICE_URL}/session/{session_id}/send-media",
                    json={
                        "to": phone,
                        "media": base64_img,
                        "caption": text
                    },
                    timeout=40.0
                )

                if 400 <= response.status_code < 500:
                    if "Session not connected" in response.text:
                        logger.warning("Session disconnected. Retrying…")
                        await asyncio.sleep(2)
                        continue

                    raise HTTPException(status_code=response.status_code, detail=response.text)

                response.raise_for_status()
                logger.info(f"📸 Image message sent to {phone}")
                return response.json()

            except Exception as e:
                last_exception = e
                logger.warning(f"[IMAGE] Failed on attempt {attempt}: {str(e)}")

                if attempt < max_retries:
                    await asyncio.sleep(2)

    raise HTTPException(status_code=504, detail=f"Failed to send image message: {str(last_exception)}")

if __name__ == "__main__":
    # Test the functions here if needed
    org_id = "fc442d0b-17a5-479c-8cda-4bd0a30c0c5e"
    fetch_rest_detail(org_id)
    print("---")