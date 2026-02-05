# 🍍 Leaving a pineapple by the door as requested.
import json
import logging
from contextlib import asynccontextmanager
from datetime import date
from typing import List, Literal, Optional
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import statistics

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fx_builder")

FRANKFURTER_API_URL = "https://api.frankfurter.dev"
FALLBACK_FILE = Path("data/sample_fx.json")

# --- Models ---
class DailyRate(BaseModel):
    date: str
    rate: float
    pct_change: Optional[float] = None

class SummaryResponse(BaseModel):
    start_rate: float
    end_rate: float
    total_pct_change: float
    mean_rate: float
    breakdown: Optional[List[DailyRate]] = None

# --- Helpers ---
def calculate_pct_change(current: float, previous: float) -> float:
    """Safely calculates percentage change guarding against division by zero."""
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100

async def fetch_fx_rates(start_date: date, end_date: date, from_curr: str, to_curr: str):
    """
    Fetches rates from Frankfurter API. 
    Implements a 'shield of protection' (simple retry logic).
    """
    url = f"{FRANKFURTER_API_URL}/{start_date}..{end_date}"
    params = {"from": from_curr, "to": to_curr}
    
    async with httpx.AsyncClient() as client:
        try:
            # Simple shield: 1 retry on connection error
            for attempt in range(2):
                try:
                    resp = await client.get(url, params=params, timeout=5.0)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("rates", {})
                except (httpx.ConnectError, httpx.TimeoutException):
                    if attempt == 1: raise
                    logger.warning("Connection failed, retrying...")
        except Exception as e:
            logger.error(f"Network failed: {e}. Attempting fallback.")
            return load_fallback_data(start_date, end_date, to_curr)

def load_fallback_data(start_date: date, end_date: date, target_curr: str):
    """Fallback to local JSON if the network fails."""
    if not FALLBACK_FILE.exists():
        logger.error("Fallback file not found.")
        return {}
    
    try:
        with open(FALLBACK_FILE, "r") as f:
            data = json.load(f)
            # Filter logically (simplified for this exercise)
            rates = data.get("rates", {})
            return {k: v for k, v in rates.items() if str(start_date) <= k <= str(end_date)}
    except Exception as e:
        logger.error(f"Failed to load fallback: {e}")
        return {}

# --- App Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Titanium FX Engine...")
    yield
    logger.info("Shutting down.")

app = FastAPI(title="FX Intelligence Layer", lifespan=lifespan)

# --- Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "builder": "ready"}

@app.get("/summary", response_model=SummaryResponse)
async def get_summary(
    start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
    breakdown: Literal['day', 'none'] = 'none'
):
    # 1. Fetch Data
    raw_rates = await fetch_fx_rates(start_date, end_date, "EUR", "USD")
    
    if not raw_rates:
        raise HTTPException(status_code=404, detail="No data available for this range")

    # 2. Process Data
    # Sort dates to ensure correct order
    sorted_dates = sorted(raw_rates.keys())
    
    if not sorted_dates:
         raise HTTPException(status_code=404, detail="No data found in range")

    daily_data = []
    rates_list = []
    
    previous_rate = None

    for d in sorted_dates:
        # Frankfurter returns { "USD": 1.23 }
        val = raw_rates[d]
        rate_val = val.get("USD") if isinstance(val, dict) else val
        
        pct = 0.0
        if previous_rate is not None:
            pct = calculate_pct_change(rate_val, previous_rate)
        
        daily_data.append(DailyRate(date=d, rate=rate_val, pct_change=pct))
        rates_list.append(rate_val)
        previous_rate = rate_val

    # 3. Aggregates
    start_rate = rates_list[0]
    end_rate = rates_list[-1]
    total_pct = calculate_pct_change(end_rate, start_rate)
    mean_rate = statistics.mean(rates_list)

    response = SummaryResponse(
        start_rate=start_rate,
        end_rate=end_rate,
        total_pct_change=total_pct,
        mean_rate=mean_rate,
        breakdown=daily_data if breakdown == 'day' else None
    )
    
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
