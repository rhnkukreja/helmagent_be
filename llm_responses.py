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



async def generate_followup_message(message_list):
    """
    Generate a contextual WhatsApp follow-up message based on previous AI messages and customer order data
    """
    try:
        print("\n" + "="*60)
        print("🤖 GENERATING FOLLOW-UP MESSAGE")
        
        prompt = f"""
            You are a restaurant manager at The Corner Cafe replying to a customer's recent WhatsApp message.

            Conversation History:
            {message_list}

            Your task:
            1. Read the entire conversation and understand the tone, especially the latest customer message.
            2. Analyze the tone to detect the customer's mood:
            - If they sound happy, satisfied, thankful, or positive → mood = GOOD
            - If they sound disappointed, mention poor service, bad food, delay, or complain → mood = BAD

            Now follow these rules based on the detected mood:

            🔹 If mood = GOOD:
            - Address the person as "Sir" or "Ma'am" (never use their name).
            - Write a short, warm thank-you message for their kind words.
            - Politely ask them to share their experience on Google Reviews and include a link placeholder like:
                👉 https://maps.app.goo.gl/yPpFNAig6KkrN3YdA
            - Also generate 2–3 short sample reviews (each ≤ 30 words) that they can post directly.
            - Keep tone cheerful, appreciative, and authentic (use 1–2 emojis max).
            - End politely (e.g., “Looking forward to serving you again soon!”).

            🔹 If mood = BAD:
            - Address the person as "Sir" or "Ma'am" (never use their name).
            - Gently apologize for their unpleasant experience.
            - Acknowledge what went wrong (e.g., food quality, service delay, or general disappointment).
            - Offer them a 30% discount on their next visit and encourage them to give you another chance.
            - Keep tone empathetic, sincere, and caring (1 emoji max).
            - End politely (e.g., “We truly hope to make your next visit delightful.”).

            ⚙️ Additional Context-Aware Behavior:
            - Read the conversation carefully before replying.
            - If the customer has *already acknowledged or agreed to post a review* (e.g., messages like “surely will do that”, “already did”, “posted”, “done”, or similar),
              then do NOT repeat the Google Review link or sample reviews.
              Instead, simply thank them warmly for their support and express appreciation (1 emoji max).
            - The reply should feel natural and context-aware — avoid repetition or robotic tone.

            Formatting:
            - Keep the entire message between 80–120 words.
            - Return only the WhatsApp message text.
            - Do not explain or label the mood.
            - The message should sound natural, like a real restaurant manager writing it personally.
            3. Never return any refunds or compensation other than the 30% discount for BAD mood.

            """
                  # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly restaurant manager following up with customers on WhatsApp. Write human-like, engaging, and caring messages that sound personal."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8,
            max_tokens=300
        )

        ai_message = response.choices[0].message.content.strip()

        print(f"\n✅ Follow-up AI Message Generated:")
        print(ai_message)
        print("="*60 + "\n")

        return ai_message

    except Exception as e:
        print(f"\n❌ Error generating follow-up message: {str(e)}")
        return JSONResponse(
            content={
                "success": False,
                "error": f"Failed to generate follow-up message: {str(e)}"
            },
            status_code=500
        )

