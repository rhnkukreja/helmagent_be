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

# -------------------- Configuration --------------------
# Load environment variables or hardcode for testing
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-project.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-supabase-service-role-key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
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
    Uses GPT-4 Vision to extract structured data (name, contact_number, items_ordered)
    from an image of a bill or invoice.
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
                            "text": """Extract the following information strictly in JSON:
                            {
                                "name": "<Customer Name>",
                                "contact_number": "<Mobile Number>",
                                "items_ordered": [
                                    {"item_name": "<Item>", "quantity": "<Qty>", "price": "<Price>"}
                                ]
                            }.
                            If data is missing, leave the field empty. Do not add explanations.""",
                        },
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_bytes.decode()}},
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
    Stores extracted data into the Supabase table.
    """
    try:
        print("Storing extracted data in Supabase...")
        record = {
            "name": extracted_data.get("name", ""),
            "contact_number": extracted_data.get("contact_number", ""),
            "items_ordered": json.dumps(extracted_data.get("items_ordered", [])),
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
):
    """
    Processes a bill either by uploaded image OR by image URL.
    Returns the extracted JSON and stores it in Supabase.
    """
    try:
        if not image and not image_url:
            raise HTTPException(status_code=400, detail="Provide either an image file or image_url.")

        if image:
            image_bytes = await image.read()
            import base64
            encoded = base64.b64encode(image_bytes)
            extracted_data = extract_text_from_image(encoded)
        else:
            extracted_data = extract_text_from_url(image_url)

        stored_data = store_in_supabase(extracted_data)
        return JSONResponse(content={"status": "success", "data": stored_data})

    except Exception as e:
        print("Error in /process-bill/:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Root Endpoint --------------------

@app.get("/")
def root():
    return {"message": "Bill Processing API is running", "version": "1.0"}
