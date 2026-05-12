![](assets/0.jpg)

# Infinite Terrain Generator

This project focuses on generating realistic terrains with infinite resolution using Python. By leveraging Numba, the generator achieves high computational efficiency, enabling detailed exploration from large-scale continents to small valleys.

It creates a base map using diamond-square algorithm. And combines interpolation of the base map and finer frequency FBM noise to generate zoomable maps. 

---

## Gallery: Zooming In


| Level | Resolution | Asset |
|-------|----------------|-------|
| 1 - Zone | ~100 km | ![Level 1](assets/0.jpg) |
| 2 - County | ~50 km | ![Level 2](assets/1.jpg) |
| 3 - Cluster | ~25 km | ![Level 3](assets/2.jpg) |
| 4 - City | ~12 km | ![Level 4](assets/3.jpg) |
| 5 - Town | ~6 km | ![Level 5](assets/4.jpg) |
| 6 - Village | ~3 km | ![Level 6](assets/5.jpg) |
| 7 - Hamlet | ~1.5 km | ![Level 7](assets/6.jpg) |

---

## Features

- **Infinite terrains**: Generate heightmaps at any zoom level.
- **Realistic erosion**: Simulate thermal, hydraulic and wind erosion for natural-looking landscapes.
- **Customizable**: Adjust parameters to create diverse and unique terrains.
- **Visualization tools**: Preview terrains in both 2D and 3D.
- **Optimized performance**: Critical computations are accelerated using Numba.

---

## Getting Started

### Installation

```bash
git clone https://github.com/your-username/infinite-terrain-generator.git
cd infinite-terrain-generator
pip install -r requirements.txt
```

### Usage

```python
from terrain_lod import Terrain

# Create a terrain instance
t = Terrain(seed=42, erode=True)

# Generate a heightmap
t.plot(lim=(0.0, 1.0, 0.0, 1.0), zlim=(0, 3), save_path="world.png")
```

For additional examples, refer to [`example.ipynb`](example.ipynb).

---

## Project Structure

```
map_maker/
├── terrain_lod/
│   ├── terrain.py       # Core terrain generation logic
│   ├── erosion.py       # Erosion simulation algorithms
│   ├── noise.py         # Noise generation functions
│   └── helper.py        # Supporting utilities
├── assets/              # Sample images
├── example.ipynb        # Interactive notebook
└── README.md
```

---

## License

This project is licensed under the MIT License. You are free to use and modify it as needed.
