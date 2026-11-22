"""Request / response schemas for the prediction API.

Every field has a sensible default, so clients may send only the fields they
care about (e.g. `{"lead_time": 200, "deposit_type": "Non Refund"}`).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Month = Literal["January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"]


class Booking(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {
            "hotel": "City Hotel", "lead_time": 120, "arrival_date_month": "August",
            "stays_in_weekend_nights": 1, "stays_in_week_nights": 3,
            "adults": 2, "children": 0, "babies": 0, "meal": "BB",
            "market_segment": "Online TA", "distribution_channel": "TA/TO",
            "is_repeated_guest": 0, "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0, "reserved_room_type": "A",
            "booking_changes": 0, "deposit_type": "No Deposit",
            "days_in_waiting_list": 0, "customer_type": "Transient", "adr": 110.0,
            "required_car_parking_spaces": 0, "total_of_special_requests": 1,
        }
    })

    hotel: Literal["City Hotel", "Resort Hotel"] = "City Hotel"
    lead_time: int = Field(60, ge=0, le=800, description="Days between booking and arrival")
    arrival_date_month: Month = "August"
    stays_in_weekend_nights: int = Field(1, ge=0, le=30)
    stays_in_week_nights: int = Field(2, ge=0, le=60)
    adults: int = Field(2, ge=0, le=10)
    children: int = Field(0, ge=0, le=10)
    babies: int = Field(0, ge=0, le=10)
    meal: Literal["BB", "HB", "FB", "SC"] = "BB"
    market_segment: str = "Online TA"
    distribution_channel: str = "TA/TO"
    is_repeated_guest: int = Field(0, ge=0, le=1)
    previous_cancellations: int = Field(0, ge=0, le=50)
    previous_bookings_not_canceled: int = Field(0, ge=0, le=100)
    reserved_room_type: str = "A"
    booking_changes: int = Field(0, ge=0, le=30)
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"] = "No Deposit"
    days_in_waiting_list: int = Field(0, ge=0, le=400)
    customer_type: Literal["Transient", "Transient-Party", "Contract", "Group"] = "Transient"
    adr: float = Field(100.0, ge=0, le=1000, description="Average daily rate (EUR)")
    required_car_parking_spaces: int = Field(0, ge=0, le=8)
    total_of_special_requests: int = Field(0, ge=0, le=5)
    # Only used if the model was trained with features.include_country=true
    country: str = "Unknown"


class Prediction(BaseModel):
    cancel_probability: float
    will_cancel: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    bookings: list[Booking] = Field(..., min_length=1, max_length=500)


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
