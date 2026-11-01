from sqlalchemy import Column, Float, Integer, String, TIMESTAMP

from db.database import Base


class Vehicule(Base):
    __tablename__ = "vehicule"

    id = Column(Integer, primary_key=True, index=True)
    immatriculation = Column(String, index=True, nullable=False)
    
    statut = Column(String, nullable=False, default="safe")


class Detection(Base):
    __tablename__ = "detection"

    id = Column(Integer, primary_key=True, index=True)
    immatriculation = Column(String, index=True, nullable=False)
    confiance = Column(Float, nullable=False)
    detected_at = Column(TIMESTAMP, nullable=False)
    source_type = Column(String, nullable=False)
    camera_id = Column(Integer, nullable=True)


class Alert(Base):
    __tablename__ = "alert"

    id = Column(Integer, primary_key=True, index=True)
    immatriculation = Column(String, index=True, nullable=False)
    infraction = Column(String, nullable=False)
    temps = Column(TIMESTAMP, nullable=False)
