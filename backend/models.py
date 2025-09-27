from sqlalchemy import Column, Integer, String, ForeignKey, TIMESTAMP, LargeBinary, func
from sqlalchemy.orm import relationship
from database import Base

class Batch(Base):
    __tablename__ = "batches"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    filter = Column(String(50))
    created_at = Column(TIMESTAMP, server_default=func.now())

    images = relationship("Image", back_populates="batch", cascade="all, delete")

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("batches.id", ondelete="CASCADE"))
    filename = Column(String(255), nullable=False)
    filter = Column(String(50))
    processed = Column(LargeBinary)  # store processed image as binary
    created_at = Column(TIMESTAMP, server_default=func.now())

    batch = relationship("Batch", back_populates="images")
