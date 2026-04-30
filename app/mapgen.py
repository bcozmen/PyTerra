import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import heapq
from scipy.ndimage import distance_transform_edt
class World():
    def __init__(self, size = 2**9+1, sea_level  = 0.18, 
                    num_rivers = 5, 
                    seed = 43, diamond_square_parameters = {}):
        if seed is not None:
                np.random.seed(seed)

        self.size = size
        self.num_rivers = num_rivers
        self.sea_level = sea_level
        self.diamond_square_parameters = diamond_square_parameters

        self.generate_heightmap()
        self.generate_sea_mask()


        self.generate_rivers()
        self.generate_fertility_map()
        self.generate_forest_map()

    def plot_2d(self, map_type = "terrain"):
        h = self.heightmap
        sea = self.sea_mask | self.river_mask

        plt.figure(figsize=(10, 10))
        
        if map_type == "terrain":
            plt.imshow(h, cmap="terrain", vmin=0, vmax=1.)
        elif map_type == "fertility":
            plt.imshow(self.fertility_map, cmap="Greens", vmin=0, vmax=1.)
        elif map_type == "forest":
            plt.imshow(self.forest_map, cmap="Greens", vmin=0, vmax=1.)

        overlay = np.zeros((*h.shape, 4))
        overlay[..., 2] = sea
        overlay[..., 3] = sea * 0.5
        plt.imshow(overlay)

        #colorbar
        
        plt.title(f"Final {map_type.capitalize()}: Sea Level = {self.sea_level}")
        plt.show()

    
    def generate_heightmap(self):
        scale = self.diamond_square_parameters.get('scale', 1.0)
        roughness = self.diamond_square_parameters.get('roughness', 0.45)
        assert (self.size - 1) & (self.size - 2) == 0, "size must be 2^n + 1"

        grid = np.zeros((self.size, self.size), dtype=np.float32)

        grid[0, 0] = np.random.rand() * scale
        grid[0, -1] = np.random.rand() * scale
        grid[-1, 0] = np.random.rand() * scale
        grid[-1, -1] = np.random.rand() * scale

        step = self.size - 1
        current = scale

        while step > 1:
            half = step // 2

            # square step
            for x in range(half, self.size, step):
                for y in range(half, self.size, step):
                    grid[x, y] = (
                        grid[x-half, y-half] +
                        grid[x-half, y+half] +
                        grid[x+half, y-half] +
                        grid[x+half, y+half]
                    ) / 4.0 + (np.random.rand() - 0.5) * current

            # diamond step
            for x in range(0, self.size, half):
                for y in range((x + half) % step, self.size, step):
                    vals = []
                    if x - half >= 0: vals.append(grid[x-half, y])
                    if x + half < self.size: vals.append(grid[x+half, y])
                    if y - half >= 0: vals.append(grid[x, y-half])
                    if y + half < self.size: vals.append(grid[x, y+half])

                    grid[x, y] = np.mean(vals) + (np.random.rand() - 0.5) * current

            step //= 2
            current *= roughness

        grid = gaussian_filter(grid, sigma=1.0)
        self.heightmap = grid

    def generate_fertility_map(self):
        #fertility is higher near rivers and low to mid altitudes

        river_effect = np.zeros_like(self.heightmap, dtype=np.float32)

        #Give a boost to fertility based on proximity to rivers
        
        river_distance = distance_transform_edt(~self.river_mask)
        river_effect += np.exp(-river_distance / self.size)  # decay with distance

        #normalize fertility to [0, 1]
        river_effect = (river_effect - river_effect.min()) / (river_effect.max() - river_effect.min() + 1e-5)

        # Apply altitude effect: boost fertility at low to mid altitudes, reduce at high altitudes
        altitude = np.clip(self.heightmap - self.sea_level, 0, 1)
        altitude_effect = np.exp(-((altitude) ** 2) / (2 * 0.25 ** 2))  # Gaussian centered at 0.5
        river_effect *= altitude_effect

        #normalize again after altitude effect
        river_effect[self.sea_mask | self.river_mask] = 0  # sea and river have zero fertility
        river_effect = (river_effect - river_effect.min()) / (river_effect.max() - river_effect.min() + 1e-5)
        self.fertility_map = river_effect

        
    def generate_forest_map(self):
        #forest is more exponentially more likely in higer altitudes
        height = np.clip(self.heightmap - self.sea_level, 0, 1)
        altitude_effect = np.exp(height)  # exponential boost for higher altitudes
        normalized_altitude_effect = (altitude_effect - altitude_effect.min()) / (altitude_effect.max() - altitude_effect.min() + 1e-5)

        self.forest_map = normalized_altitude_effect + normalized_altitude_effect * self.fertility_map
        self.forest_map = (self.forest_map - self.forest_map.min()) / (self.forest_map.max() - self.forest_map.min() + 1e-5)
    def generate_sea_mask(self):
        self.sea_mask = self.heightmap < self.sea_level
        self.river_mask = np.zeros_like(self.sea_mask, dtype=bool)
        self.river_count = np.zeros_like(self.sea_mask, dtype=np.int32)

    def generate_rivers(self):
        #get the %10 of the highest points that are not sea as river sources and randomly pick num_rivers of them as river sources
        h, w = self.heightmap.shape
        #choose from top 50 and probabilistically sample according to height (higher altitudes more likely)
        land_mask = ~self.sea_mask
        land_heights = self.heightmap[land_mask]
        threshold = np.percentile(land_heights, 93)
        potential_sources = np.argwhere((self.heightmap >= threshold) & land_mask)
        np.random.shuffle(potential_sources)
        sources = potential_sources[:self.num_rivers]
        # river_count tracks how many rivers pass through each cell
        self.river_count = np.zeros((h, w), dtype=np.int32)

        for start_x, start_y in sources:
            self.generate_river(start_x, start_y)

        # Widen rivers proportionally: cells where N rivers merge are dilated N times,
        # so downstream merged rivers are progressively wider than their tributaries.
        from scipy.ndimage import binary_dilation
        max_count = int(self.river_count.max())
        dilated = np.zeros((h, w), dtype=bool)
        for level in range(1, max_count + 1):
            # Every cell carrying at least `level` rivers contributes a dilation of `level` iterations
            mask = self.river_count >= level
            dilated |= binary_dilation(mask, iterations=level)

        self.river_mask = dilated
    def generate_river(self, start_x, start_y):
        h, w = self.heightmap.shape

        # cost map
        cost = np.full((h, w), np.inf)
        cost[start_x, start_y] = 0

        # parent pointers for path reconstruction
        parent = {}

        # priority queue: (cost, x, y)
        pq = [(0, start_x, start_y)]

        while pq:
            c, x, y = heapq.heappop(pq)

            # reached sea → stop
            if self.sea_mask[x, y]:
                end = (x, y)
                break

            # skip outdated entries
            if c > cost[x, y]:
                continue

            current_h = self.heightmap[x, y]

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = x + dx, y + dy

                    if not (0 <= nx < h and 0 <= ny < w):
                        continue

                    nh = self.heightmap[nx, ny]

                    # uphill penalty (key idea)
                    step_cost = max(0, nh - current_h)

                    new_cost = c + step_cost

                    if new_cost < cost[nx, ny]:
                        cost[nx, ny] = new_cost
                        parent[(nx, ny)] = (x, y)
                        heapq.heappush(pq, (new_cost, nx, ny))
        else:
            return None  # no path found

        # reconstruct path
        path = []
        cur = end
        while cur != (start_x, start_y):
            path.append(cur)
            cur = parent[cur]
        path.append((start_x, start_y))
        path.reverse()

        # build river count map (increment for each river passing through)
        for x, y in path:
            self.river_count[x, y] += 1

        # keep river_mask in sync so downstream code / sea checks stay valid
        self.river_mask = self.river_count > 0


world = World(size=2**8+1, num_rivers=8, sea_level=0.18, seed=43)

world.plot_2d()
world.plot_2d("fertility")
world.plot_2d("forest")