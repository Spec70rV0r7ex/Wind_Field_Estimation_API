from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from .gee_utils import init_gee
from .wind_retrieval import retrieve_wind_field
from .validation import calculate_metrics

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_gee()
    yield

app = FastAPI(
    title="SAR Ocean Wind Retrieval API", 
    description="Sentinel-1 CMOD5.N High-Resolution Wind Inversion",
    lifespan=lifespan
)

class WindRequest(BaseModel):
    # Expects a series of [Longitude, Latitude] pairs forming a polygon
    polygon: List[List[float]]  
    date: str          
    scale: int = 5000  

@app.post("/retrieve_winds")
def get_winds(req: WindRequest):
    try:
        # Basic validation to ensure they passed a valid polygon (at least 3 points)
        if len(req.polygon) < 3:
            raise ValueError("A polygon requires a series of at least 3 coordinate pairs.")

        data = retrieve_wind_field(req.polygon, req.date, req.scale)
        if not data:
            return {"status": "success", "message": "No S1 imagery found on this date for this geometry.", "data": []}
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
def validate_winds(req: WindRequest):
    try:
        if len(req.polygon) < 3:
            raise ValueError("A polygon requires a series of at least 3 coordinate pairs.")
            
        data = retrieve_wind_field(req.polygon, req.date, req.scale)
        if not data:
             return {"status": "success", "message": "No S1 imagery found on this date for this geometry.", "metrics": {}}
        metrics = calculate_metrics(data)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
