# --- bill OCR dependencies ---
# Add at top 3rd algorithm file main_new.py
# --- bill OCR dependencies ---
import base64
import os
import io
import json
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv


# Load the .env file
load_dotenv()

# -------------------- Configuration --------------------
# Load environment variables
SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment variables
if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing required environment variables. Check your .env file")

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
def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_json_response(raw_output: str) -> str:
    """
    Removes markdown code blocks and extracts clean JSON from GPT response
    """
    # Remove markdown code blocks (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r'^```(?:json)?\s*\n', '', raw_output.strip())
    cleaned = re.sub(r'\n```$', '', cleaned.strip())
    return cleaned.strip()


def extract_bill_info(image_path: str) -> dict:
    """
    Reads the image file, encodes it to base64, and sends it to GPT-4 Vision for data extraction.
    """
    try:
        print("Encoding image to base64...")
        image_base64 = encode_image(image_path)
        
        print("Sending request to GPT-4 Vision...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert data extractor. Analyze the image of a restaurant bill and extract key details in structured JSON format. Return ONLY valid JSON without any markdown formatting or code blocks.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract the following information strictly in JSON format (NO markdown, NO code blocks, ONLY raw JSON):
                            {
                                "name": "<Customer Name>",
                                "contact_number": "<Mobile Number>",
                                "items_ordered": [
                                    {"item_name": "<Item>", "quantity": "<Qty>", "price": "<Price>"}
                                ]
                            }
                            If customer name or contact number is missing, leave those fields as empty strings.
                            Return ONLY the JSON object, nothing else.""",

                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                },
            ],
            temperature=0.2,
        )

        raw_output = response.choices[0].message.content
        print("Raw GPT Output:", raw_output)
        
        # Clean the response to remove markdown code blocks
        cleaned_output = clean_json_response(raw_output)
        print("Cleaned Output:", cleaned_output)
        
        # Parse JSON
        data = json.loads(cleaned_output)
        print("Successfully parsed JSON:", data)
        return data

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        print(f"Failed to parse: {raw_output}")
        raise HTTPException(status_code=500, detail=f"Failed to parse GPT response as JSON: {str(e)}")
    except Exception as e:
        print("Error in extract_bill_info:", str(e))
        raise HTTPException(status_code=500, detail=f"Bill extraction error: {str(e)}")


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

        print("Record to insert:", record)
        response = supabase.table("bills").insert(record).execute()

        if response.data:
            print("Data stored successfully:", response.data)
            return response.data
        else:
            print("Failed response:", response)
            raise Exception(f"Supabase returned no data: {response}")

    except Exception as e:
        print("Error storing in Supabase:", str(e))
        raise HTTPException(status_code=500, detail=f"Supabase error: {str(e)}")

# -------------------- API Routes --------------------

@app.post("/process-bill/")
async def process_bill(file: UploadFile = File(...)):
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        print(f"Saving uploaded file to {temp_path}...")
        
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        print("Processing bill image...")

        # Extract info
        extracted_data = extract_bill_info(temp_path)
        print("Extraction successful!")

        # Store in Supabase
        stored_data = store_in_supabase(extracted_data)
        print("Storage successful!")

        return JSONResponse(content={
            "success": True,
            "message": "Bill processed successfully",
            "extracted_data": extracted_data,
            "stored_data": stored_data
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    finally:
        # Always cleanup temp file
        if temp_path and os.path.exists(temp_path):
            print(f"Cleaning up temp file: {temp_path}")
            os.remove(temp_path)


@app.get("/")
def root():
    return {"message": "Bill Processing API is running", "version": "1.0"}


# -------------------- Run Server --------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting Bill Processor API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)