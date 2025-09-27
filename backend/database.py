from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Use psycopg2 driver for Postgres
DATABASE_URL = "postgresql+psycopg2://user:09282402911@localhost:5433/imagedb"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()
1