# --- bill OCR dependencies ---
# Add at top 3rd algorithm file main_new.py


import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pdf2image import convert_from_bytes
import cv2
import numpy as np
import io
import re
import os
import difflib
import traceback
# --- Tesseract Path (adjust for your system) ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- FastAPI Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBearer()


# --- IMAGE PREPROCESSING ---
def preprocess_image(img: Image.Image) -> Image.Image:
    """Advanced image preprocessing with adaptive binarization & deskewing"""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Denoising and adaptive thresholding
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Morphological cleanup
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Deskewing (straighten tilted text)
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = thresh.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        thresh = cv2.warpAffine(
            thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    return Image.fromarray(thresh)


# --- OCR FUNCTION ---
def ocr_from_image(img: Image.Image) -> str:
    """Improved OCR configuration for better accuracy"""
    config = (
        "--dpi 300 --oem 3 --psm 6 "
        "--tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@.-:/()+ "
    )
    text = pytesseract.image_to_string(img, config=config)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


# --- FUZZY MATCH HELPER ---
def fuzzy_match(target, candidates, cutoff=0.7):
    match = difflib.get_close_matches(target, candidates, n=1, cutoff=cutoff)
    return match[0] if match else target


# --- CUSTOMER NAME EXTRACTION ---
def extract_customer_name(text: str) -> str | None:
    lines = text.splitlines()
    name_candidates = []

    for line in lines:
        if re.search(r"(customer|name|guest|client|to)", line, re.IGNORECASE):
            name_match = re.search(
                r"(?:Name|Customer|Client|To)\s*[:\-]?\s*([A-Za-z\s\.]+)", line
            )
            if name_match:
                candidate = name_match.group(1).strip()
                if 3 <= len(candidate) <= 50:
                    name_candidates.append(candidate)

    if not name_candidates:
        for line in lines:
            if re.match(r"^(Mr|Mrs|Ms|Dr|Sri|Smt)\.?\s+[A-Za-z]+", line):
                return line.strip()

    if name_candidates:
        cleaned = fuzzy_match(
            name_candidates[0],
            [re.sub(r"[^A-Za-z\s]", "", x) for x in name_candidates],
        )
        return cleaned.strip().title()

    return None


# --- PHONE NUMBER EXTRACTION ---
def extract_phone(text: str) -> str | None:
    text = re.sub(r"[^\d+]", "", text)
    phone_match = re.search(r"(\+?91\d{10}|\b[6-9]\d{9}\b)", text)
    if phone_match:
        number = phone_match.group(1)
        if not number.startswith("+91"):
            number = "+91" + number[-10:]
        return number
    return None


# --- ITEM EXTRACTION ---
def extract_items(text: str) -> list[dict]:
    lines = text.splitlines()
    items = []

    for line in lines:
        line_clean = line.strip()
        if (
            len(line_clean) < 3
            or re.search(r"(total|gst|tax|bill|amount)", line_clean, re.IGNORECASE)
        ):
            continue

        parts = re.findall(r"[A-Za-z0-9\.\-/]+", line_clean)
        numbers = [float(x) for x in re.findall(r"\d+\.\d+|\d+", line_clean)]

        if len(numbers) >= 2 and len(parts) > 1:
            item_name = " ".join([p for p in parts if not re.match(r"\d+", p)])[:50]
            if item_name and not re.search(
                r"\b(gst|total|amount)\b", item_name, re.IGNORECASE
            ):
                items.append(
                    {
                        "name": item_name.strip(),
                        "quantity": int(numbers[0]) if numbers[0] < 50 else 1,
                        "price": numbers[-1],
                    }
                )

    return items

@app.post("/upload-bill")
async def upload_bill(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.lower().endswith(".pdf"):
            images = convert_from_bytes(contents)
        else:
            image = Image.open(io.BytesIO(contents))
            images = [image]

        extracted_text = ""
        for img in images:
            extracted_text += pytesseract.image_to_string(img)

        return {"extracted_text": extracted_text}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- MAIN ENDPOINT ---
@app.post("/upload-bill")
async def upload_bill(file: UploadFile = File(...), token: str = Depends(security)):
    try:
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized")

        if not (
            file.content_type.startswith("image/")
            or file.content_type == "application/pdf"
        ):
            raise HTTPException(status_code=400, detail="File must be image or PDF")

        contents = await file.read()
        full_text = ""
        processed_img = None

        # --- Handle PDF or Image ---
        if file.content_type == "application/pdf":
            pages = convert_from_bytes(contents)
            for page in pages:
                processed = preprocess_image(page)
                full_text += ocr_from_image(processed) + "\n"
                if not processed_img:
                    processed_img = processed
        else:
            img = Image.open(io.BytesIO(contents))
            processed_img = preprocess_image(img)
            full_text = ocr_from_image(processed_img)

        # --- Save processed image ---
        if processed_img:
            os.makedirs("processed_bills", exist_ok=True)
            save_path = f"processed_bills/{file.filename.split('.')[0]}_processed.jpeg"
            processed_img.save(save_path, "JPEG")
        else:
            save_path = None

        # --- Extract structured data ---
        customer_name = extract_customer_name(full_text)
        phone_number = extract_phone(full_text)
        food_items = extract_items(full_text)

        return {
            "success": True,
            "extracted": {
                "customer_name": customer_name or "Not Found",
                "phone_number": phone_number or "Not Found",
                "food_items": food_items,
            },
            "raw_ocr_text": full_text,
            "processed_image_path": save_path,
            "debug_info": {
                "total_items_found": len(food_items),
                "ocr_line_count": len(full_text.splitlines()),
                "file_type": file.content_type,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# --- ROOT ENDPOINT ---
@app.get("/")
async def root():
    return {"message": "Enhanced Bill OCR API Running"}


# --- APP RUNNER ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)



'''

import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter, ImageEnhance
from pdf2image import convert_from_bytes
import io
import re
import os


# Set tesseract path (Windows example; adjust for your system if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBearer()

def preprocess_image(img: Image.Image) -> Image.Image:
    """Preprocess image for better OCR results"""
    # Convert to grayscale
    img = img.convert("L")
    # Enhance contrast more aggressively
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)
    # Enhance sharpness
    sharpness = ImageEnhance.Sharpness(img)
    img = sharpness.enhance(2.0)
    # Resize for better OCR (scale up small images)
    w, h = img.size
    scale = 3 if min(w, h) < 1000 else 2
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # Apply sharpening filter
    img = img.filter(ImageFilter.SHARPEN)
    # Binary threshold with adjusted value
    img = img.point(lambda p: 255 if p > 140 else 0)
    return img

def ocr_from_image(img: Image.Image) -> str:
    """Extract text from preprocessed image with better config"""
    # Try multiple PSM modes for better accuracy
    configs = [
        "--oem 3 --psm 6",  # Assume uniform block of text
        "--oem 3 --psm 4",  # Assume single column of text
    ]
    
    best_text = ""
    for config in configs:
        text = pytesseract.image_to_string(img, config=config).strip()
        if len(text) > len(best_text):
            best_text = text
    
    return best_text

def extract_customer_name(text: str) -> str | None:
    """Extract customer name from bill text dynamically"""
    lines = text.splitlines()
    
    # Pattern 1: Check bottom of bill for common signatures (// SRI, MR., etc.)
    for line in reversed(lines[-15:]):
        line = line.strip()
        if not line or len(line) < 3:
            continue
            
        # Look for patterns like "// NAME" or "Mr./Mrs./Ms. NAME"
        patterns = [
            r'^//\s*(.+)$',
            r'^(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Sri|Smt\.?)\s+([A-Za-z\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up common suffixes
                name = re.sub(r'\b(Take\s+Away|Delivery|Pick\s*Up|Parcel)\b', '', name, flags=re.IGNORECASE).strip()
                # Validate: only letters and spaces, reasonable length
                if re.match(r'^[A-Za-z\s]+$', name) and 3 <= len(name) <= 50:
                    return name
    
    # Pattern 2: Check top 20 lines for explicit labels
    label_patterns = [
        r'(?:Customer|Name|Guest|Client|To)[\s:]+([A-Za-z\s]+)',
    ]
    
    for line in lines[:20]:
        line = line.strip()
        for pattern in label_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Validate
                if re.match(r'^[A-Za-z\s]+$', name) and 3 <= len(name) <= 50:
                    # Not a bill-related word
                    if not re.search(r'\b(Bill|Invoice|Receipt|Total|Date|Time)\b', name, re.IGNORECASE):
                        return name
    
    return None

def extract_phone(text: str) -> str | None:
    """Extract phone number from text dynamically"""
    # Comprehensive phone number patterns
    phone_patterns = [
        # With labels
        r'(?:Phone|Mobile|Contact|Tel|Ph|Cell|Mob)[\s:]*(\+?91[\s-]?\d{10})',
        r'(?:Phone|Mobile|Contact|Tel|Ph|Cell|Mob)[\s:]*(\d{10})',
        # Without labels but with country code
        r'\b(\+91[\s-]?\d{10})\b',
        r'\b(91\d{10})\b',
        # Standalone 10-digit numbers (starting with 6-9 for Indian mobile)
        r'\b([6-9]\d{9})\b',
        # With formatting like 98765-43210 or 9876543210
        r'\b(\d{5}[-\s]?\d{5})\b',
    ]
    
    for pattern in phone_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            phone = match.group(1).strip()
            # Clean up phone number - remove all non-digits except +
            phone = re.sub(r'[^\d+]', '', phone)
            
            # Validate and format
            if phone.startswith('+91') and len(phone) >= 12:
                return phone[:13]  # +91 + 10 digits
            elif phone.startswith('91') and len(phone) >= 12:
                return f"+{phone[:12]}"
            elif len(phone) == 10 and phone[0] in '6789':
                return f"+91{phone}"
    
    return None

def is_non_item_line(line: str) -> bool:
    """Check if line is NOT a food/product item"""
    line_lower = line.lower()
    
    # Skip keywords that indicate non-item lines

    skip_patterns = [
        r'\b(sub\s*total|subtotal|total|grand\s*total)\b',
        r'\b(cgst|sgst|gst|vat|tax|service|cess)\b',
        r'\b(discount|offer|coupon|promo)\b',
        r'\b(amount|payment|paid|balance|change)\b',
        r'\b(cash|card|upi|paytm|gpay|phonepe|credit|debit)\b',
        r'\b(thank|visit|welcome|regards)\b',
        r'\b(bill|invoice|receipt|token)\b',
        r'\b(date|time|fssai|gstin|cin)\b',
        r'\b(address|phone|email|website)\b',
        r'^(qty|particulars|items?|sr\.?|s\.?no\.?|rate|price|amount)$',
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, line_lower):
            return True
    
    return False

def extract_items(text: str) -> list[dict]:
    """
    Dynamically extract items from bill without predefined keywords.
    Uses structural patterns to identify item lines.
    """
    lines = text.splitlines()
    items = []
    
    in_items_section = False
    items_section_started = False
    
    for i, line in enumerate(lines):
        original_line = line
        line = line.strip()
        
        if not line or len(line) < 2:
            continue
        
        line_lower = line.lower()
        
        # Detect start of items section
        if not items_section_started:
            # Common headers for items section
            if re.search(r'\b(particulars|items?|description|qty|quantity|rate|price)\b', line_lower):
                in_items_section = True
                items_section_started = True
                continue
        
        # If no header found but we see a pattern like "ITEM 1 50 50", assume items started
        if not items_section_started:
            if re.match(r'^[A-Z][A-Za-z\s/]+\s+\d+\s+\d+\s+\d+', line):
                in_items_section = True
                items_section_started = True
        
        # Stop at totals/summary section
        if re.search(r'\b(sub\s*total|subtotal)\b', line_lower):
            break
        
        # Skip non-item lines
        if is_non_item_line(line):
            continue
        
        # Only process if we're in items section or found item-like pattern
        if not in_items_section and not items_section_started:
            continue
        
        # Extract items using structural patterns
        # Pattern 1: ITEM_NAME QTY RATE AMOUNT (most common)
        pattern1 = r'^([A-Za-z][A-Za-z0-9\s/\-\.\(\)]+?)\s+(\d+)\s+(\d+(?:\.\d{1,2})?)\s+(\d+(?:\.\d{1,2})?)\s*$'
        match = re.match(pattern1, line)
        
        if match:
            item_name = match.group(1).strip()
            quantity = int(match.group(2))
            rate = float(match.group(3))
            amount = float(match.group(4))
            
            # Validate item name (not just numbers, reasonable length)
            if len(item_name) >= 3 and not item_name.isdigit():
                items.append({
                    "name": item_name,
                    "quantity": quantity,
                    "price": amount
                })
                continue
        
        # Pattern 2: ITEM_NAME followed by numbers (more flexible)
        # Must have at least 2 numbers (quantity and amount, or rate and amount)
        numbers_in_line = re.findall(r'\d+(?:\.\d{1,2})?', line)
        
        if len(numbers_in_line) >= 2:
            # Extract item name (text before first number)
            name_match = re.match(r'^([A-Za-z][A-Za-z0-9\s/\-\.\(\)]+?)\s+\d', line)
            
            if name_match:
                item_name = name_match.group(1).strip()
                
                # Parse numbers based on count
                quantity = 1
                amount = None
                
                if len(numbers_in_line) >= 3:
                    # Format: QTY RATE AMOUNT
                    quantity = int(float(numbers_in_line[0]))
                    amount = float(numbers_in_line[-1])  # Last number is usually total
                elif len(numbers_in_line) == 2:
                    # Format: QTY AMOUNT or RATE AMOUNT
                    first_num = float(numbers_in_line[0])
                    if first_num <= 20:  # Likely a quantity
                        quantity = int(first_num)
                    amount = float(numbers_in_line[-1])
                
                # Validate
                if len(item_name) >= 3 and not item_name.isdigit() and amount is not None:
                    # Check for duplicates
                    if not any(item['name'].lower() == item_name.lower() for item in items):
                        # Final validation: not a bill keyword
                        if not is_non_item_line(item_name):
                            items.append({
                                "name": item_name,
                                "quantity": quantity,
                                "price": amount
                            })
    
    return items

@app.post("/upload-bill")
async def upload_bill(file: UploadFile = File(...), token: str = Depends(security)):
    try:
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if not (file.content_type.startswith("image/") or file.content_type == "application/pdf"):
            raise HTTPException(status_code=400, detail="File must be image or PDF")

        contents = await file.read()
        processed_img = None
        full_text = ""

        # Process PDF or Image
        if file.content_type == "application/pdf":
            pages = convert_from_bytes(contents)
            for page in pages:
                preprocessed = preprocess_image(page)
                text = ocr_from_image(preprocessed)
                full_text += text + "\n"
                if not processed_img:
                    processed_img = preprocessed
        else:
            img = Image.open(io.BytesIO(contents))
            processed_img = preprocess_image(img)
            full_text = ocr_from_image(processed_img)

        # Save processed image
        if processed_img:
            os.makedirs("processed_bills", exist_ok=True)
            processed_img.save(f"processed_bills/{file.filename.split('.')[0]}_processed.jpeg", "JPEG")

        # Extract structured data dynamically
        customer_name = extract_customer_name(full_text)
        phone_number = extract_phone(full_text)
        food_items = extract_items(full_text)

        # Structured response
        return {
            "success": True,
            "extracted": {
                "customer_name": customer_name or "Not Found",
                "phone_number": phone_number or "Not Found",
                "food_items": food_items
            },
            "raw_ocr_text": full_text,
            "processed_image_path": f"processed_bills/{file.filename.split('.')[0]}_processed.jpeg",
            "debug_info": {
                "total_items_found": len(food_items),
                "ocr_line_count": len(full_text.splitlines()),
                "file_type": file.content_type
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Bill OCR API - Dynamic Extraction Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)






from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter, ImageOps, ImageFont, ImageDraw
import pytesseract
from pdf2image import convert_from_bytes
import io
import re
import pytesseract

# If needed (Windows) set tesseract path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBearer()

# --- helper functions ---

def preprocess_image(img: Image.Image) -> Image.Image:
    """Grayscale -> increase contrast -> remove noise -> resize modestly for OCR."""
    # Convert to grayscale
    img = img.convert("L")
    # Resize (scale up small images for better OCR)
    w, h = img.size
    scale = 1
    if min(w, h) < 1000:
        scale = 2
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # Auto-contrast
    img = ImageOps.autocontrast(img)
    # Optional: sharpen
    img = img.filter(ImageFilter.SHARPEN)
    # Binarize (adaptive thresholding is better but this is simple)
    img = img.point(lambda p: 255 if p > 180 else 0)
    return img

def ocr_from_image(img: Image.Image) -> str:
    """Run pytesseract with a config tuned for invoice/receipt text."""
    config = "--oem 3 --psm 6"  # psm 6: assume a block of text; try 4 or 6 if needed
    text = pytesseract.image_to_string(img, config=config)
    return text

def extract_phone(text: str) -> str | None:
    # Broader phone patterns
    m = re.search(r"(\+?\d{1,3}[-\s]?)?(?:\(?\d{2,4}\)?[-\s]?)?\d{5,12}", text)
    return m.group(0).strip() if m else None

def looks_like_price(s: str) -> bool:
    return bool(re.search(r"\d[\d,]*[\.]?\d{0,2}$", s))

def extract_items(text: str) -> list:
    """
    Heuristic item extraction:
     - Remove known footer keywords (TOTAL, SUBTOTAL, GST, TAX, AMOUNT)
     - For each line, strip leading qty/indices and trailing prices;
       if remaining piece looks like an item name, add it.
     - Also scan for common item-list sections like 'Items', 'Description', 'Purchased'
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # unify lines to uppercase sentinel for section detection but keep originals
    joined = "\n".join(lines)
    # Remove obvious non-item lines
    footer_tokens = ["total", "subtotal", "gst", "tax", "balance", "amount", "net", "discount", "change"]
    cleaned_lines = []
    for ln in lines:
        low = ln.lower()
        if any(tok in low for tok in footer_tokens):
            continue
        cleaned_lines.append(ln)

    items = []
    # Section-based capture if there is Items: or Description:
    items_section_start = None
    for i, ln in enumerate(cleaned_lines):
        if re.match(r"(?i)^(items?|description|dish(es)?|order|ordered|purchased)[:\s]?$", ln):
            items_section_start = i + 1
            break
    scan_lines = cleaned_lines[items_section_start:] if items_section_start is not None else cleaned_lines

    # Candidate detection regexes
    # Pattern A: "ItemName   qty   price" (multiple spaces or tabs separate columns)
    # Pattern B: "qty x ItemName - price"
    # Pattern C: "1. ItemName 100.00"
    for ln in scan_lines:
        if len(ln) < 3:
            continue
        low = ln.lower()
        # skip pure numeric / invoice meta
        if re.match(r"^[\d\-\s\.\,\/:]+$", ln):
            continue
        # skip header-like lines
        if re.search(r"(invoice|bill|date|table|qty|price|amount|order no|tax|gst)", low):
            continue

        original = ln

        # Remove leading list numbers or qtys like "1.", "01", "2)"
        ln2 = re.sub(r"^[\d\.\)\-\s]+", "", original).strip()

        # If pattern "2 x Paneer Butter Masala 250.00"
        m = re.match(r"^(?:\d+\s*[xX]\s*)?(.*?)(?:\s+[-–]\s*|\s{2,}|\,\s*)?(\d[\d,\.]*)?$", ln2)
        if m:
            candidate = m.group(1).strip()
            trailing = m.group(2) or ""
        else:
            candidate = ln2
            trailing = ""

        # Strip trailing prices like " 250.00" or "₹250" or "250"
        candidate = re.sub(r"[\s\-\–\|]*[₹$]?[\d,]+(?:\.\d{1,2})?$", "", candidate).strip()

        # filter out short garbage
        if len(candidate) < 3:
            continue
        # filter lines that are obviously addresses, phone, or single digits/words like "cash"
        if re.match(r"^(cash|visa|mastercard|card|credit|debit|phone|tel|address)$", candidate.lower()):
            continue

        # final sanity: candidate should have at least two words or contain food-ish terms (pam, curry, naan, masala)
        if len(candidate.split()) >= 2 or re.search(r"(naan|masala|pizza|burger|chicken|paneer|rice|salad|pasta|soup|roll|sandwich|coke|pepsi|cola|coffee|tea)", candidate.lower()):
            items.append(candidate)

    # dedupe while preserving order
    seen = set()
    deduped = []
    for it in items:
        norm = it.lower()
        if norm not in seen:
            deduped.append(it)
            seen.add(norm)
    return deduped

# --- endpoint ---
@app.post("/upload-bill")
async def upload_bill(file: UploadFile = File(...), token: str = Depends(security)):
    try:
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized")

        if not (file.content_type.startswith("image/") or file.content_type == "application/pdf"):
            raise HTTPException(status_code=400, detail="File must be image or PDF")

        contents = await file.read()
        full_text = ""

        if file.content_type == "application/pdf":
            try:
                pages = convert_from_bytes(contents)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"PDF conversion error: {e}")
            for p in pages:
                pre = preprocess_image(p)
                full_text += ocr_from_image(pre) + "\n"
        else:
            try:
                img = Image.open(io.BytesIO(contents))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"PIL image open error: {e}")
            pre = preprocess_image(img)
            full_text = ocr_from_image(pre)

        phone = extract_phone(full_text)
        food_items = extract_items(full_text)

        return {
            "status": "success",
            "ocr_raw": full_text,
            "extracted": {"phone": phone, "food_items": food_items}
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Server error:\n{str(e)}\n{tb}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





import sys
import uvicorn
import asyncio
import logging
import gc
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import APIRouter, Request as FastAPIRequest
from fastapi.responses import JSONResponse
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from services.google_drive.drive_scanner import (
    get_authorization_url,
    exchange_code_for_credentials,
    download_drive_files,
    build
)
from services.google_drive.drive_resume_details import process_resume_files
import os
import uuid
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()
drive_router = APIRouter(tags=["drive"])


# api routes
from api.v1.routes_auth import auth_router
from api.v1.routes_outlook import outlook_router
from api.v1.routes_job import job_router
from api.v1.routes_jobadder import jobadder_router
from api.v1.routes_user import user_router
from api.v1.routes_crm import crm_router
from api.v1.routes_drive import drive_router
from api.v1.routes_linkedin import linkedin_router
from src.api.v1.routes_email_outreach_ai import email_service_router
from api.v1.routes_gpt_mode import gpt_mode_router
from src.api.v1.routes_call import call_router



# config


supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
supabase: Client = create_client(supabase_url, supabase_key)
redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")

def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False


@drive_router.get("/api/auth/drive/start")
async def start_drive_auth():
    try:
        auth_url, state, error = get_authorization_url()
        if not auth_url:
            return JSONResponse(status_code=500, content={"status": "error", "message": error})
        return JSONResponse(status_code=200, content={"status": "success", "auth_url": auth_url, "state": state})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@drive_router.get("/api/auth/drive/verify")
async def verify_and_estimate_drive_data(request: FastAPIRequest):
    try:
        auth_code = request.query_params.get('code')
        user_id = request.query_params.get('user_id')
        state = request.query_params.get('state')

        if not auth_code or not user_id:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Both 'code' and 'user_id' are required"})
        if not is_valid_uuid(user_id):
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid user_id format"})

        creds, email, error = exchange_code_for_credentials(auth_code, state)
        if error:
            return JSONResponse(status_code=400, content={"status": "error", "message": error})

        if creds.refresh_token:
            supabase.table("user_crm_details").upsert(
                {"user_id": user_id, "google_refresh_token": creds.refresh_token},
                on_conflict=["user_id"]
            ).execute()

        service = build('drive', 'v3', credentials=creds)
        results = service.files().list(
            pageSize=1000,
            q="mimeType != 'application/vnd.google-apps.folder'",
            fields="files(id, name, mimeType)"
        ).execute()
        files_metadata = results.get('files', [])
        total_files = len(files_metadata)
        estimated_time_sec = total_files * 2.5
        estimated_minutes = round(estimated_time_sec / 60, 2)

        return JSONResponse(status_code=200, content={
            "status": "success",
            "total_files_detected": total_files,
            "estimated_time_min": estimated_minutes,
            
        })

    except Exception as e:
        logger.exception("Verify failed")
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Internal error: {str(e)}"})


@drive_router.post("/api/drive/fetch")
async def fetch_drive_data_from_refresh_token(request: FastAPIRequest):
    try:
        data = await request.json()
        user_id = data.get("user_id")
        if not user_id or not is_valid_uuid(user_id):
            return JSONResponse(status_code=400, content={"status": "error", "message": "Valid 'user_id' is required"})

        response = supabase.table("user_crm_details").select("google_refresh_token, email").eq("user_id", user_id).execute()
        record = response.data[0] if response.data else None
        if not record or not record.get("google_refresh_token"):
            return JSONResponse(status_code=404, content={"status": "error", "message": "No refresh token found for user"})

        refresh_token = record["google_refresh_token"]
        email = record.get("email", "unknown_user@gmail.com")

        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET
        )
        creds.refresh(GoogleRequest())

        downloaded_files, total_files, file_type_counts, document_files, error = download_drive_files(creds, email)
        if downloaded_files is None:
            return JSONResponse(status_code=500, content={"status": "error", "message": error})

        document_results, missing_details = process_resume_files(document_files, email, user_id)
        for doc in document_results:
            try:
                supabase.table("gdrive_candidate_details").insert(doc).execute()
                logger.info(f"Saved resume_id {doc['resume_id']} to gdrive_candidate_details")
            except Exception as e:
                logger.error(f"Error saving resume_id {doc['resume_id']}: {str(e)}")

        industries = [doc["industry"] for doc in document_results if doc["industry"]]
        industry = max(set(industries), key=industries.count) if industries else "Unknown"

        candidate_status = {"ready": 0, "warm": 0, "cold": 0}
        for doc in document_results:
            if all([doc.get("email_id"), doc.get("phone_no"), doc.get("education"), doc.get("tech_skills"), doc.get("work_experience")]):
                candidate_status["ready"] += 1
            elif doc.get("email_id") and doc.get("tech_skills"):
                candidate_status["warm"] += 1
            else:
                candidate_status["cold"] += 1

        dashboard_data = {
            "total_files": total_files,
            "file_type_counts": file_type_counts,
            "missing_details": missing_details,
            "industry": industry,
            "candidate_status": candidate_status
        }

        return JSONResponse(status_code=200, content={
            "status": "success",
            "user_id": user_id,
            "dashboard_data": dashboard_data
        })

    except Exception as e:
        logger.exception("Fetch failed")
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Internal error: {str(e)}"})
    
    
# ✅ Run GC at startup
@app.on_event("startup")
async def startup_event():
    gc.collect()
    logging.info("Garbage collector run at startup to free up memory.")

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",    # Your current frontend
        "http://localhost:8081",    # Your current frontend
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8000",
        "https://resourcerai.netlify.app",
        "https://resourcerai-staging.netlify.app",
        "https://resourcer-ai-2.onrender.com",
        "http://172.20.10.6:8080",  # Your network address
        FRONTEND_DOMAIN_URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# api routes
app.include_router(auth_router, prefix="/auth")
#app.include_router(job_router, prefix="/job")
app.include_router(user_router, prefix="/user")
#app.include_router(jobadder_router, prefix="/jobadder")
app.include_router(drive_router, prefix="/drive")
#app.include_router(outlook_router, prefix="/outlook")
#app.include_router(crm_router, prefix="/crm")
#app.include_router(linkedin_router, prefix="/linkedin")
app.include_router(email_service_router, prefix="/send_email")
#app.include_router(gpt_mode_router, prefix="/gpt-mode")
app.include_router(call_router, prefix="/call", tags=["call"])



@app.get("/")
def read_root():
    return {"message": "Welcome to the Recruiter AI APP Backend"}


if __name__ == "__main__":
    uvicorn.run("main_app:app", host="0.0.0.0", port=8000, reload=True)


'''