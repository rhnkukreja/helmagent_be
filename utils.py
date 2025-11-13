import os 
import json
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi import HTTPException
load_dotenv()

SUPABASE_KEY=os.getenv("SUPABASE_KEY")
SUPABASE_URL=os.getenv("SUPABASE_URL")

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