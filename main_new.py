# --- bill OCR dependencies ---
# Add at top 3rd algorithm file main_new.py

import os
import io
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv
# -------------------- Configuration --------------------
# Load environment variables or hardcode for testing
load_dotenv()

SUPABASE_KEY=os.getenv("SUPABASE_KEY")
SUPABASE_URL=os.getenv("SUPABASE_URL")

supabase: Client = create_client(SUPABASE_URL,SUPABASE_KEY)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Bill Processor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Helper Functions --------------------

import json
from fastapi import HTTPException

def extract_text_from_image(image_bytes: bytes) -> dict:
    """
    Uses GPT-4 Vision to extract structured data (name, contact_number, items_ordered, date, total_amount)
    from an image of a bill or invoice.
    Returns dict with keys: name, contact_number, items_ordered (list), date (YYYY-MM-DD), total_amount (float).
    """
    try:
        print("Processing image with GPT-4 Vision...")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert data extractor. Analyze the image of a restaurant bill and extract key details in structured JSON format.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract the following fields from the bill image in strict JSON format:
                            {
                            "name": "<Customer Name>",
                            "contact_number": "<Customer Mobile Number>",
                            "items_ordered": [
                                {"item_name": "<Item>", "quantity": "<Qty>", "price": "<Price>"}
                            ],
                            "date": "<Bill date in ISO YYYY-MM-DD>",
                            "total_amount": "<Numeric Total>"
                            }

                            ⚠️ Important rules:
                            - Ignore restaurant or merchant names, phone numbers, GST numbers, invoice numbers, and cashier names.
                            - Only extract the *customer's* name and phone number if explicitly shown (like “Customer Name”, “Bill To”, or “Contact No”).
                            - If no customer name or contact number is present, leave those fields empty.
                            - Always include `items_ordered`, `date`, and `total_amount` if visible.
                            - Return ONLY JSON — no extra text."""
                        },
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_bytes.decode()}},
                    ],
                },
            ],
            temperature=0.1,
        )

        raw_output = response.choices[0].message.content.strip()
        print("Raw GPT Output:", raw_output)

        # Clean up any markdown-style JSON fences (just in case)
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`")
            if "json" in raw_output:
                raw_output = raw_output.replace("json", "", 1).strip()
        
        result = json.loads(raw_output)
        print("Extracted Data:", result)

        return result

    except Exception as e:
        print("Error extracting text:", str(e))
        raise HTTPException(status_code=500, detail=f"GPT-4 Vision error: {str(e)}")


def store_in_supabase(extracted_data: dict):
    """
    Stores extracted data into the Supabase table, now with bill_date and total_amount.
    """
    try:
        print("Storing extracted data in Supabase...")
        record = {
            "name": extracted_data.get("name", ""),
            "contact_number": extracted_data.get("contact_number", ""),
            "items_ordered": json.dumps(extracted_data.get("items_ordered", [])),
            # expect date as 'YYYY-MM-DD' or empty string
            "bill_date": extracted_data.get("date") or None,
            # store numeric total - if missing, store None/0.0 depending on your preference
            "total_amount": extracted_data.get("total_amount") if extracted_data.get("total_amount") not in (None, "", 0) else None,
            # keep org_id if present
            "org_id": extracted_data.get("org_id", None),
        }

        response = supabase.table("bills").insert(record).execute()

        if response.data:
            print("Data stored successfully:", response.data)
            return response.data
        else:
            raise Exception(str(response))

    except Exception as e:
        print("Error storing in Supabase:", str(e))
        raise HTTPException(status_code=500, detail=f"Supabase error: {str(e)}")


# -------------------- API Routes --------------------


@app.post("/process-bill/")
async def process_bill(
    image: UploadFile = File(None),
    image_url: str = Form(None),
    org_id: str = Form(None),
):
    """
    Processes a bill either by uploaded image OR by image URL.
    Returns the extracted JSON and stores it in Supabase.
    """
    try:
        if not image and not image_url:
            raise HTTPException(status_code=400, detail="Provide either an image file or image_url.")
        if not org_id:
            raise HTTPException(status_code=400, detail="Missing organization ID (org_id).")

        if image:
            import base64
            image_bytes = await image.read()
            encoded = base64.b64encode(image_bytes)
            extracted_data = extract_text_from_image(encoded)
        else:
            extracted_data = extract_text_from_url(image_url)

        # Include org_id in data before storing
        extracted_data["org_id"] = org_id

        stored_data = store_in_supabase(extracted_data)
        return JSONResponse(content={"status": "success", "data": stored_data})

    except Exception as e:
        print("Error in /process-bill/:", str(e))
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/dashboard/{org_id}")
async def get_dashboard_data(org_id: str):
    try:
        print("Fetching organization for org_id:", org_id)

        # --- fetch safely ---
        query = (
            supabase.table("organizations")
            .select("*")
            .eq("org_id", org_id)  # Supabase client should handle UUID conversion
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

from datetime import datetime

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


# --- ROOT ENDPOINT ---
@app.get("/")
async def root():
    return {"message": "Enhanced Bill OCR API Running"}

# --- APP RUNNER ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_new:app", host="0.0.0.0", port=8000, reload=True)


