from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Statut(str, Enum):
    SAFE = "safe"
    VOLE = "vole"
    SUSPECT= "suspicieux"
    INCONNU = "inconnu"


class SourceType(str, Enum):
    VIDEO = "video"
    PHOTO = "photo"


class Vehicule(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    immatriculation: str = Field(min_length=1)
    # proprietaire: str
    # marque: str
    # couleur: str
    statut: Statut


class Detection(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    immatriculation: str = Field(min_length=1)
    confiance: float = Field(ge=0, le=1)
    detected_at: datetime
    source_type: SourceType
    camera_id: int | None = None


class Alert(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    immatriculation: str = Field(min_length=1)
    infraction: Statut
    temps: datetime
