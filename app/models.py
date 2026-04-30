from typing import List, Optional, Literal, Union
from pydantic import BaseModel


class Position(BaseModel):
    """Continuous 2D position"""
    x: float
    y: float


class RenderConfig(BaseModel):
    """Visual rendering configuration for a place"""
    fillColor: str = "#cccccc"
    strokeColor: str = "#333333"
    strokeWidth: float = 1.0
    opacity: float = 1.0
    zIndex: int = 0  # Lower renders first (background)


class Place(BaseModel):
    """
    Unified model for all geographic/architectural features.
    Can represent mountains, forests, farms, buildings, rivers, roads, etc.
    """
    id: int
    name: str
    type: Literal[
        "mountain", "forest", "farm", "building", 
        "river", "road", "water", "terrain"
    ]
    
    # Geometry - either polygon or polyline
    polygon: Optional[List[Position]] = None  # For area features
    points: Optional[List[Position]] = None   # For line features
    
    # Type-specific attributes
    attributes: dict = {}  # e.g., {"crop_type": "wheat", "density": 0.8, "building_type": "shop"}
    
    # Visual rendering
    render: RenderConfig = RenderConfig()


class Entity(BaseModel):
    """Dynamic entities like people, animals, vehicles"""
    id: int
    name: str
    pos: Position
    wealth: float = 0.0
    happiness: float = 1.0
    
    # Rendering config for entities
    render: Optional[dict] = None  # {"color": "#ff6b35", "radius": 6, "zIndex": 100}


class WorldState(BaseModel):
    tick: int
    width: float
    height: float
    places: List[Place]  # All geographic/architectural features
    entities: List[Entity]  # All dynamic entities
