![](assets/0.png)

# Infinite Terrain Generator


Welcome to the Infinite Terrain Generator! This project is all about creating realistic, infinite-resolution terrains using Python. Whether you're zooming out to see entire continents or zooming in to explore tiny valleys, this generator has you covered.

---

## Gallery: Zooming In

Here’s a sneak peek at what the generator can do. Each image shows the same world, zoomed in progressively:


| Level | Length | Asset |
|-------|----------------:|-------|
| Level 1 — World | 100 km | ![Level 1](assets/0.png) |
| Level 2 — Continent | 50 km | ![Level 2](assets/1.png) |
| Level 3 — Region | 20 km | ![Level 3](assets/2.png) |
| Level 4 — Area | 10 km | ![Level 4](assets/3.png) |
| Level 5 — Valley | 5 km | ![Level 5](assets/4.png) |

---

## Features

- **Infinite terrains**: Generate heightmaps at any zoom level without seams.
- **Realistic erosion**: Simulate thermal and hydraulic erosion for natural-looking landscapes.
- **Customizable**: Tweak parameters to create unique worlds.
- **Built-in visualization**: Quickly preview your terrains in 2D and 3D.

---

## How It Works

The generator starts with a Diamond-Square fractal base, adds layers of noise for detail, and applies erosion to make it look natural. The result? A seamless, infinite terrain that looks like it could exist in the real world.

---

## Getting Started

### Installation

```bash
git clone https://github.com/your-username/infinite-terrain-generator.git
cd infinite-terrain-generator
pip install numpy scipy matplotlib numba
```

### Usage

```python
from terrain_lod import Terrain

# Create a terrain instance
t = Terrain(seed=42, erode=True)

# Generate a heightmap
t.plot(lim=(0.0, 1.0, 0.0, 1.0), zlim=(0, 3), save_path="world.png")
```

For more examples, check out [`example.ipynb`](example.ipynb).

---

## Project Structure

```
map_maker/
├── terrain_lod/
│   ├── terrain.py       # Main terrain generation logic
│   ├── erosion.py       # Erosion simulations
│   ├── noise.py         # Noise functions
│   └── helper.py        # Utility functions
├── assets/              # Example images
├── example.ipynb        # Interactive notebook
└── README.md
```

---

## License

This project is licensed under the MIT License. Feel free to use and modify it as you like!
