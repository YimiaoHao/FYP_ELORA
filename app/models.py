from sqlalchemy import Column, Integer, String, Float, Date, DateTime
from sqlalchemy.sql import func
from .database import Base


class Record(Base):
    __tablename__ = "records"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    height_m = Column(Float, nullable=False)
    weight_kg = Column(Float, nullable=False)
    family_history = Column(String(1), nullable=False)   # Y/N
    activity_level = Column(String(10), nullable=False)  # low/medium/high
    water_ml = Column(Integer, nullable=False)
    bmi = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
