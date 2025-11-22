import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.responses import JSONResponse


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# 🧠 Extract Text from Image (Async)
# ============================================================
async def extract_text_from_image(image_base64: str) -> dict:
    """
    Uses GPT-5 Vision to extract structured data (name, contact_number, items_ordered, date, total_amount)
    from an image of a bill or invoice.
    Returns dict with keys: name, contact_number, items_ordered (list), date (YYYY-MM-DD), total_amount (float).
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-5-nano",
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
                            - If there is any currency symbol (₹, $, etc.) in prices, dont remove it — keep it as is.
                            - Always include `items_ordered`, `date`, and `total_amount` if visible.
                            - Return ONLY JSON — no extra text."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        },
                    ],
                },
            ],
            temperature=1,
        )

        raw_output = response.choices[0].message.content.strip()

        # Clean up any markdown-style JSON fences
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`").replace("json", "", 1).strip()

        result = json.loads(raw_output)
       
        return result

    except Exception as e:
        print("❌ Error extracting text from image:", str(e))
        raise HTTPException(status_code=500, detail=f"GPT-4 Vision error: {str(e)}")


# ============================================================
# 🌐 Extract Text from HTML (Async)
# ============================================================
async def extract_text_from_html(html_content: str) -> dict:
    """
    Uses GPT-5 to extract structured data (name, contact_number, items_ordered, date, total_amount)
    from an uploaded HTML bill file.
    """
    try:
        print("Processing HTML with GPT-5...")

        response = await client.chat.completions.create(
            model="gpt-5-nano",
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
                            - If there is any currency symbol (₹, $, etc.) in prices, dont remove it — keep it as is.
                            - If any field is missing, leave it blank or empty.
                            - Output ONLY pure JSON (no markdown, no commentary)."""
                        },
                        {"type": "text", "text": html_content},
                    ],
                },
            ],
            temperature=1,
        )

        raw_output = response.choices[0].message.content.strip()

        # Clean up markdown fences
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`").replace("json", "", 1).strip()

        result = json.loads(raw_output)
        return result

    except Exception as e:
        print("❌ Error extracting text from HTML:", str(e))
        raise HTTPException(status_code=500, detail=f"GPT-4 HTML extraction error: {str(e)}")


# ============================================================
# 💬 Generate WhatsApp Follow-up Message (Async)
# ============================================================
async def generate_followup_message(message_list, restaurant_name, google_review_link):
    """Generate contextual WhatsApp follow-up with correct Sir/Ma'am."""
    try:
        print("\n" + "=" * 60)
        print("🤖 GENERATING FOLLOW-UP MESSAGE")

        # Detect gender
        gender = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": f"Is customer male/female? Respond: male/female/unknown\n\n{message_list}"}],
            max_completion_tokens=5,
        )
        salutation = "Ma'am" if "female" in gender.choices[0].message.content.lower() else "Sir"
        
        # Generate response
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"You're a manager at {restaurant_name}. Use only '{salutation}'. Be warm, human-like. 80-120 words."
                },
                {
                    "role": "user",
                    "content": f"""Conversation: {message_list}


If the user's latest reply is "1", "2", "one","two" (case-insensitive and irrespctive of language), then:
→ Do NOT generate a new conversation reply.
→ Simply return the exact sample review text that corresponds to that number from the previous message.
→ Output ONLY that text. No extra words, no salutation, no sign-off, nothing else.

Otherwise follow the normal rules:

If GOOD mood → thank them, ask for review and give this review link ({google_review_link}),  also give 2 sample reviews
If BAD mood → apologize, offer 30% discount
If NEUTRAL → answer their question

Formatting requirements (MUST follow exactly):
For GOOD mood include a short numbered list of reviews corresponding exactly to what they've had and reviews increase restaurant's reputation:
   1.) “First sample review”
   2.) “Second sample review”
   Then add this line (exactly): 
   Select any one number, i will share that with you and you can copy paste

Use '{salutation}' only. No names. One message only.
"""
                },
            ],
            temperature=0.85,
            max_tokens=250,
        )
        
        ai_message = response.choices[0].message.content.strip()
        print(f"\n✅ Message: {ai_message}\n" + "=" * 60 + "\n")
        return ai_message
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)




if __name__=="__main__":
    message_list=["""Hello Prateek Sir,

Thank you so much for dining with us at Timshel on October 11, 2025! We hope you enjoyed our flavorful Kadhai Chicken, prepared with a blend of aromatic spices that truly brings out its essence, along with the comforting steam rice. 🍚

We’d love to hear about your experience! Your feedback is invaluable to us as we strive to make every visit memorable.

Looking forward to your thoughts!

Warm regards,
Timshel Team""",
"Thank you",
"""Thank you, Sir! We appreciate your feedback and would love to hear your thoughts on your experience. If you could take a moment to leave a review here: ABCD_link, it would mean a lot to us. 

Here are two sample reviews you might consider:  
1.) “The Kadhai Chicken was bursting with flavor and cooked to perfection! A must-try!”  
2.) “The ambiance was delightful, and the steam rice complemented the dish wonderfully.”  

Select any one number (Type 1 or 2), I will share that with you and you can copy paste.
""",
"2"
]

    restaurant_name="ABCD"
    google_review_link="ABCD_link"
    import asyncio

    print(asyncio.run(generate_followup_message(message_list, restaurant_name, google_review_link)))



