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

def extract_text_from_image(image_bytes: bytes) -> dict:
    """
    Uses GPT-4 Vision to extract structured data (name, contact_number, items_ordered, date, total_amount)
    from an image of a bill or invoice.
    Returns dict with keys: name, contact_number, items_ordered (list), date (YYYY-MM-DD or ""), total_amount (float or 0.0)
    """
    try:
        import re
        from datetime import datetime

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
                            "text": """Extract the following information strictly in JSON:
{
  "name": "<Customer Name>",
  "contact_number": "<Mobile Number>",
  "items_ordered": [
    {"item_name": "<Item>", "quantity": "<Qty>", "price": "<Price>"}
  ],
  "date": "<Bill date in ISO YYYY-MM-DD if possible>",
  "total_amount": "<Total numeric amount, e.g., 123.45>"
}
If a field is missing, return an empty string or 0 for amounts. DO NOT add explanatory text — return only JSON or a code block containing JSON."""
                        },
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_bytes.decode()}},
                    ],
                },
            ],
            temperature=0.1,
        )

        raw_output = response.choices[0].message.content
        print("Raw GPT Output:", raw_output)

        # --- sanitize GPT output from markdown/code fences ---
        clean_output = raw_output.strip()
        clean_output = re.sub(r"^```(?:json)?", "", clean_output, flags=re.IGNORECASE)
        clean_output = re.sub(r"```$", "", clean_output)
        clean_output = clean_output.strip()

        # Try to parse JSON directly
        parsed = {}
        try:
            parsed = json.loads(clean_output)
        except Exception as parse_err:
            # If direct parse fails, attempt to extract a JSON substring
            print("json.loads failed, trying to extract JSON substring:", parse_err)
            match = re.search(r"(\{[\s\S]*\})", clean_output)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except Exception as parse_err2:
                    print("Second json.loads attempt failed:", parse_err2)
                    parsed = {}
            else:
                parsed = {}

        # Ensure keys exist
        name = parsed.get("name", "") if isinstance(parsed.get("name", ""), str) else ""
        contact_number = parsed.get("contact_number", "") if isinstance(parsed.get("contact_number", ""), str) else ""
        items_ordered = parsed.get("items_ordered", []) if isinstance(parsed.get("items_ordered", []), list) else []

        # --- DATE extraction/normalization ---
        bill_date = ""
        raw_date_candidates = []

        # 1) prefer parsed date if present
        if parsed.get("date"):
            raw_date_candidates.append(str(parsed.get("date")))

        # 2) fallback: try to find date-like tokens in the cleaned output
        # common date patterns: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, MMM DD, YYYY, etc.
        date_patterns = [
            r"\b(\d{4}-\d{2}-\d{2})\b",
            r"\b(\d{2}/\d{2}/\d{4})\b",
            r"\b(\d{2}-\d{2}-\d{4})\b",
            r"\b(\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s?,?\s?\d{4})\b",
            r"\b(?:on\s)?(\d{1,2}/\d{1,2}/\d{2,4})\b",
        ]
        for p in date_patterns:
            for m in re.findall(p, clean_output, flags=re.IGNORECASE):
                if m:
                    raw_date_candidates.append(m)

        # Try to parse candidates with common formats
        def try_parse_date(s: str):
            formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%d %b %Y", "%d %B %Y"]
            for fmt in formats:
                try:
                    dt = datetime.strptime(s.strip(), fmt)
                    return dt.date().isoformat()
                except Exception:
                    continue
            # try a relaxed numeric-only YYYYMMDD
            s_clean = re.sub(r"[^\d]", "", s)
            if len(s_clean) == 8:
                try:
                    dt = datetime.strptime(s_clean, "%Y%m%d")
                    return dt.date().isoformat()
                except Exception:
                    pass
            return None

        for candidate in raw_date_candidates:
            parsed_date = try_parse_date(candidate)
            if parsed_date:
                bill_date = parsed_date
                break

        # --- TOTAL extraction/normalization ---
        total_amount = 0.0
        # 1) prefer parsed total_amount if present
        if parsed.get("total_amount") not in (None, "", []):
            try:
                total_amount = float(str(parsed.get("total_amount")).replace(",", "").strip())
            except Exception:
                total_amount = 0.0

        # 2) fallback: regex search for total-like lines in the cleaned output
        if not total_amount or total_amount == 0.0:
            # look for lines containing total/grand total/amount payable, etc.
            total_regexes = [
                r"total(?:\s+amount)?[:\s]*₹?\s*([0-9]+(?:[.,][0-9]{1,2})?)",
                r"grand total[:\s]*₹?\s*([0-9]+(?:[.,][0-9]{1,2})?)",
                r"amount payable[:\s]*₹?\s*([0-9]+(?:[.,][0-9]{1,2})?)",
                r"amount[:\s]*₹?\s*([0-9]+(?:[.,][0-9]{1,2})?)\s*$",
                r"₹\s*([0-9]+(?:[.,][0-9]{1,2})?)",
            ]
            for rx in total_regexes:
                m = re.search(rx, clean_output, flags=re.IGNORECASE | re.MULTILINE)
                if m:
                    try:
                        total_amount = float(m.group(1).replace(",", "").strip())
                        break
                    except Exception:
                        continue

        # Final sanitize: ensure types
        try:
            total_amount = float(total_amount)
        except Exception:
            total_amount = 0.0

        # Build result dict
        result = {
            "name": name,
            "contact_number": contact_number,
            "items_ordered": items_ordered,
            "date": bill_date,
            "total_amount": total_amount,
        }

        print("Extracted data:", result)
        return result

    except Exception as e:
        print("Error extracting text:", str(e))
        raise HTTPException(status_code=500, detail=f"GPT-4 Vision error: {str(e)}")



def extract_text_from_url(image_url: str) -> dict:
    """
    Extracts structured data from a remote image URL.
    """
    try:
        print("Processing image URL with GPT-4 Vision...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert data extractor for invoices and receipts.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """Extract the following JSON:
                        {
                            "name": "<Customer Name>",
                            "contact_number": "<Mobile Number>",
                            "items_ordered": [
                                {"item_name": "<Item>", "quantity": "<Qty>", "price": "<Price>"}
                            ]
                        }.
                        Do not include explanations or text outside JSON."""},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=0.1,
        )

        raw_output = response.choices[0].message.content
        print("Raw GPT Output:", raw_output)
        data = json.loads(raw_output)
        return data

    except Exception as e:
        print("Error extracting text from URL:", str(e))
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


