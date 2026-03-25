from sqlalchemy import Column, Integer, Float, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Program(Base):
    __tablename__ = "programs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    program = Column(Text)
    megacluster = Column(Text)
    megacluster_desc = Column(Text)
    megacluster_desc_big = Column(Text)
    institute = Column(Text)
    institute_desc = Column(Text)
    institute_desc_big = Column(Text)
    major = Column(Text)
    major_def = Column(Text)
    major_desc = Column(Text)
    major_desc_big = Column(Text)
    tracks = Column(Text)
    qual = Column(Text)
    edu_form = Column(Text)
    edu_years = Column(Float)
    pass_2024 = Column(Float)
    budget_2025 = Column(Integer)
    budget_2025_common = Column(Integer)
    contract_2025 = Column(Integer)
    contract_2025_common = Column(Integer)
    desc = Column(Text)
    skills = Column(Text)
    cost = Column(Float)
    eges_contract = Column(Text)
    eges_budget = Column(Text)
