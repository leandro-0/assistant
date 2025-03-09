from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
from app.config.general import DB_URL

engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

Base.metadata.create_all(engine)
