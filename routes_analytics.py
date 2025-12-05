import os
import json
import ast
import re
from fastapi import APIRouter, HTTPException
from supabase import create_client, Client
import pandas as pd
from datetime import datetime
import logging

# ============= CONFIGURATION & INITIALIZATION =============

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client | None = None
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables are missing.")
else:
    try:
        # Note: If running locally, ensure SUPABASE_SERVICE_KEY is set for RLS bypass
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY) 
        logger.info("Supabase client initialized.")
    except Exception as e:
        logger.error(f"Error creating Supabase client: {e}")


# ============= HELPERS =============

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize data cleaning:
    1. Creates the 'amount' column from 'total_amount' (fixing the KeyError).
    2. Ensures 'bill_date' is a proper datetime object.
    """
    if df.empty:
        return df
    
    # 1. Clean total_amount: REMOVE currency symbols/commas, then convert to numeric
    if 'total_amount' in df.columns:
        # Regex to strip everything except digits and decimal point
        df['amount'] = df['total_amount'].astype(str).str.replace(r'[^\d.]', '', regex=True) 
        df['amount'] = pd.to_numeric(df['amount'], errors="coerce").fillna(0)
    else:
        # Fallback to prevent crash if 'total_amount' is somehow missing
        logger.warning("Column 'total_amount' not found in fetched data. Setting 'amount' to 0.")
        df['amount'] = 0.0
    
    # 2. Convert bill_date to datetime and drop invalid dates
    if 'bill_date' in df.columns:
        # errors='coerce' turns bad dates into NaT (Not a Time)
        df['bill_date'] = pd.to_datetime(df['bill_date'], errors="coerce", utc=True)
        df = df.dropna(subset=['bill_date']) 

    return df

def parse_items(items_ordered_str: str) -> list:
    """Safely parses the string representation of JSON array for items_ordered."""
    if not items_ordered_str:
        return []
    
    try:
        # Try JSON parsing first (standard for modern data)
        data = json.loads(items_ordered_str) 
    except:
        try:
            # Fall back to literal evaluation (for Python string representations)
            data = ast.literal_eval(items_ordered_str) 
        except:
            # Final fallback
            return []
    
    cleaned_items = []
    for item in data:
        if isinstance(item, dict):
            try:
                qty_str = str(item.get('quantity', 0)).replace(',', '')
                quantity = float(qty_str)
                cleaned_items.append({'item_name': item.get('item_name'), 'quantity': quantity})
            except:
                continue
    return cleaned_items

def fetch_bills(org_id: str) -> pd.DataFrame:
    """Fetch bills data from Supabase and return a CLEANED DataFrame."""
    if supabase is None:
        logger.error("Supabase client is not initialized. Check environment variables.")
        return pd.DataFrame([])

    try:
        logger.info(f"Fetching bills for organization ID: {org_id}")
        # Selecting only the columns strictly needed
        res = (
            supabase.table("bills")
            .select("id, name, items_ordered, bill_date, total_amount, status, order_type")
            .eq("org_id", org_id)
            .execute()
        )
        data_count = len(res.data) if res.data else 0
        logger.info(f"Successfully fetched {data_count} bills.")
        
        df = pd.DataFrame(res.data) if res.data else pd.DataFrame([])
        
        # *** CRUCIAL STEP THAT CREATES THE 'amount' COLUMN ***
        return clean_dataframe(df) 
        
    except Exception as e:
        logger.error(f"Supabase fetch error for org {org_id}: {e}", exc_info=True)
        return pd.DataFrame([])

# ============= ENDPOINTS =============

@router.get("/{org_id}/summary")
def get_summary(org_id: str):
    """Total revenue, orders, avg order value"""
    df = fetch_bills(org_id)
    if df.empty:
        return {"total_orders": 0, "total_revenue": 0.0, "aov": 0.0}

    total_orders = len(df)
    total_revenue = df["amount"].sum() 
    aov = total_revenue / total_orders if total_orders else 0.0

    return {
        "total_orders": total_orders,
        "total_revenue": round(total_revenue, 2),
        "aov": round(aov, 2),
    }

@router.get("/{org_id}/daily-trend")
def daily_trend(org_id: str):
    """Daily revenue and AOV trend"""
    df = fetch_bills(org_id)
    if df.empty:
        return []
    
    df['date'] = df['bill_date'].dt.strftime('%Y-%m-%d')
    daily_agg = df.groupby('date')['amount'].agg(['sum', 'mean']).reset_index()
    
    daily_trend_data = daily_agg.rename(columns={'sum': 'revenue', 'mean': 'aov'}).to_dict(orient='records')
    
    return daily_trend_data

@router.get("/{org_id}/top-items")
def top_items(org_id: str):
    """Top 10 selling items by quantity"""
    df = fetch_bills(org_id)
    if df.empty:
        return []
    
    all_items = []
    for items_list in df['items_ordered']:
        all_items.extend(parse_items(items_list))

    df_items = pd.DataFrame(all_items)
    if df_items.empty:
        return []
        
    top_items = df_items.groupby('item_name')['quantity'].sum().sort_values(ascending=False).head(10).reset_index()
    
    return top_items.rename(columns={'item_name': 'name', 'quantity': 'qty'}).to_dict(orient='records')

@router.get("/{org_id}/top-customers")
def top_customers(org_id: str):
    """Top 10 customers by total spend"""
    df = fetch_bills(org_id)
    if df.empty:
        return []
    
    top_customers = df.groupby('name')['amount'].sum().sort_values(ascending=False).head(10).reset_index()
    
    return top_customers.rename(columns={'amount': 'spend'}).to_dict(orient='records')

@router.get("/{org_id}/order-types")
def order_types(org_id: str):
    """Order type distribution (e.g., Delivery, Dining, Takeaway)"""
    df = fetch_bills(org_id)
    if df.empty:
        return []
    
    # Filter out records where 'order_type' is null or an empty string
    df_filtered = df[df['order_type'].notna() & (df['order_type'].astype(str).str.strip() != '')].copy()
    
    if df_filtered.empty:
        # Returns an empty list if no valid data is found
        return [] 
    
    # Convert all to lowercase for consistent grouping
    df_filtered['order_type'] = df_filtered["order_type"].str.lower()
    
    # Calculate counts and prepare the list of dictionaries
    counts = df_filtered["order_type"].value_counts().reset_index()
    
    # Rename columns to the required 'name' and 'value'
    counts.columns = ['name', 'value']
    
    # Returns Array of Objects: [{"name": "delivery", "value": 10}, ...]
    return counts.to_dict(orient='records')

@router.get("/{org_id}/order-status")
def order_status(org_id: str):
    """Order status distribution (based on 'status' boolean column)"""
    df = fetch_bills(org_id)
    if df.empty:
        return []

    # Returns 'name' and 'value' keys for the PieChart
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['status_bool', 'value']
    
    status_counts['name'] = status_counts['status_bool'].apply(
        lambda x: 'Completed/Paid' if x is True else 'Pending/Cancelled'
    )
    
    return status_counts[['name', 'value']].to_dict(orient='records')

@router.get("/{org_id}/peak")
def peak_data(org_id: str):
    """Peak day and hour analysis based on order count."""
    df = fetch_bills(org_id)
    if df.empty:
        return {"peak_day": {"day": "N/A", "count": 0}, "peak_hour_trend": []}

    df["day"] = df["bill_date"].dt.day_name()
    df["hour"] = df["bill_date"].dt.hour.fillna(0).astype(int) 

    # --- Peak Day Logic ---
    day_counts = df["day"].value_counts()
    peak_day = {
        "day": day_counts.idxmax(),
        "count": int(day_counts.max()) 
    } if not day_counts.empty else {"day": "N/A", "count": 0}

    # --- Peak Hour Trend Logic (for the chart) ---
    hour_counts = df["hour"].value_counts().sort_index()
    
    # Ensure all 24 hours (0-23) are present, filling missing hours with 0 orders
    full_hour_counts = pd.Series(0, index=range(24)).add(hour_counts, fill_value=0)
    
    peak_hour_data = [
        {"hour": f"{h:02d}:00", "count": int(count)} # Format hour as HH:00
        for h, count in full_hour_counts.items()
    ]
    
    return {
        "peak_day": peak_day,
        "peak_hour_trend": peak_hour_data, # Array of 24 hours
    }