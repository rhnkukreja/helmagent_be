import os
import json
import ast
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
                              "order_type: "<Dine-in/Takeaway/Delivery>",
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
        classified_gender = gender.choices[0].message.content.lower()
        print("classified gender:", classified_gender)

        if "female" in classified_gender:
            salutation = "Ma'am"
        elif "male" in classified_gender:
            salutation = "Sir"
        else:
            # If the response is 'unknown' or anything else not explicitly 'male' or 'female'
            salutation = "Guest"
        print("Using salutation:", salutation)
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
        #print(f"\n✅ Message: {ai_message}\n" + "=" * 60 + "\n")
        print("✅ Follow-up message generated successfully.")
        print("=" * 60 + "\n")
        print(ai_message)
        print("=" * 60 + "\n")
        return ai_message
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

async def generate_flux_prompt(raw_dict: dict) -> str:
    system_prompt = (
        "Rewrite restaurant promotional data into a professional banner prompt. "
        "The output must force the image model to create a promotional poster layout, "
        "not a plain food photo. "
        "Rules: Include only very short text such as the title and discount. "
        "Specify placement: title at top, price at bottom. "
        "Never include long sentences or descriptions as text. "
        "Include these words exactly as short text elements."
    )

    user_prompt = (
        f"Raw Data: {raw_dict}\n"
        "Generate a 2-line promotional poster prompt with layout instructions."
    )

    try:
        response = await client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract the LLM output correctly
        output = response.output[0].content[0].text
        print("LLM OUTPUT:", output)

        return output.strip()

    except Exception as e:
        print("LLM ERROR:", e)
        return f"ERROR generating banner prompt: {str(e)}"




async def generate_offer_message(
    customer: dict,
    insights: dict,
    active_offers: list[dict],
) -> tuple[str, dict]:
    """Send active offers + customer data to LLM and let it decide. 
    If no match found → fallback to Flat 10% OFF."""
    
    try:
        print("entered generate_offer_message")

        # SAFETY: If no active offers from DB → fallback directly
        if not active_offers:
            fallback_offer = {
                "title": "Flat 10% OFF",
                "description": "Get 10% OFF on your next order — today only!"
            }
            return (
                "Hey! 🎉 Enjoy a flat 10% OFF on your next order. Valid today only!",
                fallback_offer
            )

        # 1. BUILD PROMPT
        prompt = f"""
        You are a restaurant marketing assistant.

        Your job:
        - Analyse the customer data below.
        - Select EXACTLY ONE offer from the list of ACTIVE_OFFERS.
        - If none logically match the customer's ordering behaviour, choose `"fallback"`.

        CUSTOMER:
        - Name: {customer.get("name")}
        - Most Ordered Item: {insights.get("most_ordered_item")}
        - Latest Ordered Item: {insights.get("latest_item")}

        ACTIVE_OFFERS (JSON list):
        {active_offers}

        RULES FOR OFFER SELECTION:
            - The offer MUST come from ACTIVE_OFFERS only.
            - Match based on relevance to items ordered or customer patterns.
            - If NO offer matches, respond with: fallback
            - Respond with ONLY one of these:
            - The exact offer JSON object (MUST USE DOUBLE QUOTES key="value")
            - The word: fallback

        Now respond with ONLY the selected offer (JSON or "fallback"), nothing else.
        """

        # 2. LLM CALL TO SELECT THE OFFER
        offer_choice = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
        )

        raw_offer = offer_choice.choices[0].message.content.strip()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(raw_offer)    
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # 3. HANDLE FALLBACK OFFER
        if raw_offer.lower() == "fallback":
            selected_offer = {
                "title": "Flat 10% OFF",
                "description": "Get 10% OFF on your next order — today only!"
            }
        else:   
            try:
                selected_offer = json.loads(raw_offer)
            except json.JSONDecodeError:
                try:
                    # FALLBACK TRY: Parse as Python Dictionary (Single quotes)
                    # This fixes the specific error you are seeing in logs
                    selected_offer = ast.literal_eval(raw_offer)
                except Exception as e:
                    print(f"❌ Parsing Error: {e} | Raw: {raw_offer}")
                    # Final safety fallback
                    selected_offer = {
                        "title": "Flat 10% OFF",
                        "description": "Get 10% OFF on your next order — today only!"
                    }

        # 4. BUILD FINAL MESSAGE PROMPT
        message_prompt = f"""
        Create a short WhatsApp promotional message.

        CUSTOMER:
        - Name: {customer.get("name")}

        OFFER TO PROMOTE:
        {selected_offer}

        RULES:
        - 2–3 lines max
        - Friendly WhatsApp style
        - use the desc of the offer to create a catchy message 
        - give the code  that may be in the selected offer
        - Max 1–2 emojis
        - Add urgency (“today only”, “limited time”)
        - Return ONLY the message text.
        """

        llm_response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": message_prompt}],
        )

        final_message = llm_response.choices[0].message.content.strip()
        return final_message, selected_offer

    except Exception as e:
        print("LLM Error:", e)
        fallback_offer = {
            "title": "Flat 10% OFF",
            "description": "Get 10% OFF on your next order — today only!"
        }
        return "⚠️ Unable to generate message right now. Please try again.", fallback_offer










if __name__=="__main__":
    message_list=["""
Hello Dear Guest, \n\nThank you so much for choosing Himani for your delivery on November 6, 2025! We hope you enjoyed our delicious Garlic Naan and the rich flavors of our Kadhai Paneer and Dal Makhni. Our chef takes great pride in crafting these dishes with fresh ingredients and traditional spices. \n\nWe would love to hear your thoughts about your experience! Your feedback is invaluable to us and helps us serve you better in the future. \n\nLooking forward to your reply! \n\nWarm regards,  \nHimani Team ❤️
  },
""",
"Thanks i really loved the food!"
]

    restaurant_name="ABCD"
    google_review_link="ABCD_link"
    import asyncio

    print(asyncio.run(generate_followup_message(message_list, restaurant_name, google_review_link)))



