import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


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

def extract_text_from_html(html_content: str) -> dict:
    """
    Uses GPT-4 to extract structured data (name, contact_number, items_ordered, date, total_amount)
    from an uploaded HTML bill file.
    """
    try:
        print("Processing HTML with GPT-4...")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert parser of billing data. "
                        "Analyze the provided HTML document of a restaurant bill "
                        "and extract the data in structured JSON format."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract the following fields from the HTML bill:
                            {
                              "name": "<Customer Name>",
                              "contact_number": "<Customer Mobile Number>",
                              "items_ordered": [
                                {"item_name": "<Item>", "quantity": "<Qty>", "price": "<Price>"}
                              ],
                              "date": "<Bill date in ISO YYYY-MM-DD>",
                              "total_amount": "<Numeric Total>"
                            }

                            ⚠️ Rules:
                            - Ignore restaurant info, GST, invoice no., and cashier name.
                            - Focus only on customer details, items, date, and total.
                            - If any field is missing, leave it blank or empty.
                            - Output ONLY pure JSON (no markdown, no commentary)."""
                        },
                        {
                            "type": "text",
                            "text": html_content,
                        },
                    ],
                },
            ],
            temperature=0.1,
        )

        raw_output = response.choices[0].message.content.strip()
        print("Raw GPT Output (HTML):", raw_output)

        # 🧹 Clean up markdown fences if GPT returns ```json blocks
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`")
            if "json" in raw_output:
                raw_output = raw_output.replace("json", "", 1).strip()

        result = json.loads(raw_output)
        print("Extracted HTML Data:", result)

        return result

    except Exception as e:
        print("Error extracting text from HTML:", str(e))
        raise HTTPException(status_code=500, detail=f"GPT-4 HTML extraction error: {str(e)}")

