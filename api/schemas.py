# api/schemas.py
from pydantic import BaseModel, Field
from typing import List
import datetime

class WindRequest(BaseModel):
    date: datetime.date = Field(..., description="Date of interest (YYYY-MM-DD)")
    polygon: List[List[float]] = Field(..., description="Coast of Gujarat coordinates [[lon, lat], ...]")
    
class VectorPoint(BaseModel):
    lat: float
    lon: float
    u: float
    v: float
    speed: float
    direction: float
    true_speed: float | None = None
    true_dir: float | None = None

class WindResponse(BaseModel):
    date: datetime.date
    vectors: List[VectorPoint]