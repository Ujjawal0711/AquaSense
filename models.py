from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from database import Base

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    dataset_type = Column(String, nullable=False, index=True)
    dataset_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False)

    records = relationship("Record", back_populates="dataset", cascade="all, delete-orphan")

class Record(Base):
    __tablename__ = "records"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), index=True, nullable=False)
    data = Column(JSON, nullable=False)

    dataset = relationship("Dataset", back_populates="records")
