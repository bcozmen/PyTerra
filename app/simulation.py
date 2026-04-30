from typing import List
import random
import math
from .models import WorldState, Entity, Position, Place, RenderConfig


class Simulation:
    def __init__(self, width: float = 800.0, height: float = 600.0, seed: int = None):
        self.width = width
        self.height = height
        self.tick = 0
        self.random = random.Random(seed)
        
        # Unified list of all places
        self.places: List[Place] = []
        self.entities: List[Entity] = []
        
        # ID counter for places
        self._next_place_id = 0
        
        # Generate initial world
        self._generate_world()

    def _generate_world(self):
        """Generate a small town with surrounding natural features"""
        cx, cy = self.width / 2, self.height / 2  # center
        
        # 1. Create a river running through the map
        self._create_river()
        
        # 2. Create mountains in the background
        self._create_mountains()
        
        # 3. Create forests
        self._create_forests()
        
        # 4. Create farms around the town
        self._create_farms()
        
        # 5. Create town center with buildings
        self._create_town(cx, cy)
        
        # 6. Create road network
        self._create_roads(cx, cy)
        
        # 7. Place entities (people) in the town
        self._create_entities(cx, cy)

    def _create_river(self):
        """Create a meandering river"""
        points = []
        # River flows from top to bottom with curves
        num_points = 15
        for i in range(num_points):
            y = (i / (num_points - 1)) * self.height
            # Sine wave for meandering
            x = self.width * 0.3 + 60 * math.sin(i * 0.5)
            points.append(Position(x=x, y=y))
        
        self.places.append(Place(
            id=self._next_place_id,
            name="Main River",
            type="river",
            points=points,
            attributes={"width": 12.0},
            render=RenderConfig(
                fillColor="none",
                strokeColor="#4a90d9",
                strokeWidth=12.0,
                opacity=0.7,
                zIndex=40
            )
        ))
        self._next_place_id += 1

    def _create_mountains(self):
        """Create mountain ranges"""
        # Mountain range in the top-right
        self.places.append(Place(
            id=self._next_place_id,
            name="Northern Mountains",
            type="mountain",
            polygon=[
                Position(x=self.width * 0.6, y=0),
                Position(x=self.width, y=0),
                Position(x=self.width, y=self.height * 0.4),
                Position(x=self.width * 0.7, y=self.height * 0.3),
            ],
            attributes={"height": 150.0},
            render=RenderConfig(
                fillColor="#a0826d",
                strokeColor="#8b7355",
                strokeWidth=2.0,
                opacity=1.0,
                zIndex=10
            )
        ))
        self._next_place_id += 1

    def _create_forests(self):
        """Create forest areas"""
        # Forest patch 1 - left side
        self.places.append(Place(
            id=self._next_place_id,
            name="Western Woods",
            type="forest",
            polygon=[
                Position(x=20, y=50),
                Position(x=120, y=30),
                Position(x=140, y=120),
                Position(x=60, y=140),
            ],
            attributes={"density": 0.9},
            render=RenderConfig(
                fillColor="#3d7c3d",
                strokeColor="#2d5a2d",
                strokeWidth=1.5,
                opacity=0.9,
                zIndex=20
            )
        ))
        self._next_place_id += 1
        
        # Forest patch 2 - bottom
        self.places.append(Place(
            id=self._next_place_id,
            name="Southern Woods",
            type="forest",
            polygon=[
                Position(x=self.width * 0.5, y=self.height * 0.7),
                Position(x=self.width * 0.7, y=self.height * 0.75),
                Position(x=self.width * 0.65, y=self.height * 0.95),
                Position(x=self.width * 0.45, y=self.height * 0.9),
            ],
            attributes={"density": 0.85},
            render=RenderConfig(
                fillColor="#3d7c3d",
                strokeColor="#2d5a2d",
                strokeWidth=1.5,
                opacity=0.85,
                zIndex=20
            )
        ))
        self._next_place_id += 1

    def _create_farms(self):
        """Create agricultural fields around the town"""
        # Farm 1 - north of town
        self.places.append(Place(
            id=self._next_place_id,
            name="North Farm",
            type="farm",
            polygon=[
                Position(x=self.width * 0.4, y=self.height * 0.2),
                Position(x=self.width * 0.55, y=self.height * 0.2),
                Position(x=self.width * 0.55, y=self.height * 0.35),
                Position(x=self.width * 0.4, y=self.height * 0.35),
            ],
            attributes={"crop_type": "wheat"},
            render=RenderConfig(
                fillColor="#f4e4a0",
                strokeColor="#b8a060",
                strokeWidth=1.5,
                opacity=1.0,
                zIndex=30
            )
        ))
        self._next_place_id += 1
        
        # Farm 2 - east of town
        self.places.append(Place(
            id=self._next_place_id,
            name="East Farm",
            type="farm",
            polygon=[
                Position(x=self.width * 0.65, y=self.height * 0.45),
                Position(x=self.width * 0.8, y=self.height * 0.48),
                Position(x=self.width * 0.78, y=self.height * 0.62),
                Position(x=self.width * 0.63, y=self.height * 0.6),
            ],
            attributes={"crop_type": "corn"},
            render=RenderConfig(
                fillColor="#d4c47a",
                strokeColor="#b8a060",
                strokeWidth=1.5,
                opacity=1.0,
                zIndex=30
            )
        ))
        self._next_place_id += 1

    def _create_town(self, cx: float, cy: float):
        """Create a small town with buildings"""
        
        # Town square buildings in a grid-like pattern
        for i in range(-2, 3):
            for j in range(-2, 3):
                if abs(i) + abs(j) > 3:  # Skip corners for more organic layout
                    continue
                    
                x = cx + i * 40 + self.random.uniform(-5, 5)
                y = cy + j * 40 + self.random.uniform(-5, 5)
                
                # Create rectangular building
                w = self.random.uniform(15, 25)
                h = self.random.uniform(15, 25)
                
                building_type = self.random.choice(["house", "house", "house", "shop", "warehouse"])
                
                # Building color based on type
                if building_type == "house":
                    color = "#8b4513"
                elif building_type == "shop":
                    color = "#d2691e"
                else:  # warehouse
                    color = "#a0522d"
                
                self.places.append(Place(
                    id=self._next_place_id,
                    name=f"Building_{self._next_place_id}",
                    type="building",
                    polygon=[
                        Position(x=x - w/2, y=y - h/2),
                        Position(x=x + w/2, y=y - h/2),
                        Position(x=x + w/2, y=y + h/2),
                        Position(x=x - w/2, y=y + h/2),
                    ],
                    attributes={"building_type": building_type},
                    render=RenderConfig(
                        fillColor=color,
                        strokeColor="#5a2d0c",
                        strokeWidth=2.0,
                        opacity=1.0,
                        zIndex=60
                    )
                ))
                self._next_place_id += 1

    def _create_roads(self, cx: float, cy: float):
        """Create road network connecting the town"""
        # Main horizontal road
        self.places.append(Place(
            id=self._next_place_id,
            name="Main Street East-West",
            type="road",
            points=[
                Position(x=0, y=cy),
                Position(x=cx - 80, y=cy),
                Position(x=cx + 80, y=cy),
                Position(x=self.width, y=cy),
            ],
            attributes={"width": 8.0},
            render=RenderConfig(
                fillColor="none",
                strokeColor="#6b6b6b",
                strokeWidth=8.0,
                opacity=1.0,
                zIndex=50
            )
        ))
        self._next_place_id += 1
        
        # Main vertical road
        self.places.append(Place(
            id=self._next_place_id,
            name="Main Street North-South",
            type="road",
            points=[
                Position(x=cx, y=0),
                Position(x=cx, y=cy - 80),
                Position(x=cx, y=cy + 80),
                Position(x=cx, y=self.height),
            ],
            attributes={"width": 8.0},
            render=RenderConfig(
                fillColor="none",
                strokeColor="#6b6b6b",
                strokeWidth=8.0,
                opacity=1.0,
                zIndex=50
            )
        ))
        self._next_place_id += 1
        
        # Road to farm 1
        self.places.append(Place(
            id=self._next_place_id,
            name="North Farm Road",
            type="road",
            points=[
                Position(x=cx, y=cy - 80),
                Position(x=self.width * 0.45, y=self.height * 0.25),
            ],
            attributes={"width": 5.0},
            render=RenderConfig(
                fillColor="none",
                strokeColor="#6b6b6b",
                strokeWidth=5.0,
                opacity=1.0,
                zIndex=50
            )
        ))
        self._next_place_id += 1
        
        # Road to farm 2
        self.places.append(Place(
            id=self._next_place_id,
            name="East Farm Road",
            type="road",
            points=[
                Position(x=cx + 80, y=cy),
                Position(x=self.width * 0.7, y=self.height * 0.5),
            ],
            attributes={"width": 5.0},
            render=RenderConfig(
                fillColor="none",
                strokeColor="#6b6b6b",
                strokeWidth=5.0,
                opacity=1.0,
                zIndex=50
            )
        ))
        self._next_place_id += 1

    def _create_entities(self, cx: float, cy: float):
        """Create entities (people) in the town"""
        for i in range(12):
            # Place entities near town center
            angle = self.random.random() * 2 * math.pi
            radius = self.random.uniform(10, 60)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            
            self.entities.append(Entity(
                id=i,
                name=f"Person_{i}",
                pos=Position(x=x, y=y),
                wealth=self.random.random() * 100,
                render={
                    "color": "#ff6b35",
                    "strokeColor": "#000",
                    "radius": 6,
                    "strokeWidth": 1.5,
                    "zIndex": 100
                }
            ))

    def get_state(self) -> WorldState:
        return WorldState(
            tick=self.tick,
            width=self.width,
            height=self.height,
            places=self.places,
            entities=self.entities
        )

    def tick_once(self, steps: int = 1):
        for _ in range(steps):
            self._tick()

    def _tick(self):
        """Simulate one tick - entities move in continuous space"""
        for e in self.entities:
            # Continuous random walk
            angle = self.random.random() * 2 * math.pi
            distance = self.random.uniform(0, 3.0)
            
            dx = distance * math.cos(angle)
            dy = distance * math.sin(angle)
            
            # Keep entities within bounds
            new_x = max(0, min(self.width, e.pos.x + dx))
            new_y = max(0, min(self.height, e.pos.y + dy))
            
            e.pos.x = new_x
            e.pos.y = new_y
            
            # Simple economy: small random change
            delta = self.random.uniform(-1.0, 2.0)
            e.wealth = max(0.0, e.wealth + delta)
            
            # Happiness depends on wealth
            e.happiness = max(0.0, min(1.0, 0.5 + (e.wealth / 200)))
        
        self.tick += 1
