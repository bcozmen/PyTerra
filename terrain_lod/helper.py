import numpy as np
from numba import njit



@njit(cache=True)
def diamond_square_numba(size, scale, roughness, seed=0):
    grid = np.zeros((size, size), dtype=np.float32)

    # Simple deterministic LCG RNG local to this function so behavior is reproducible
    # Uses 64-bit state
    state = np.uint64(seed if seed is not None else 0)

    def next_rand(s):
        # constants from numerical recipes
        s = np.uint64((s * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)) & np.uint64(0xFFFFFFFFFFFFFFFF))
        # convert to float in [0,1)
        return s, (s >> np.uint64(11)) / float(np.uint64(1) << np.uint64(53))

    # corners
    state, r = next_rand(state)
    grid[0, 0] = r * scale
    state, r = next_rand(state)
    grid[0, size - 1] = r * scale
    state, r = next_rand(state)
    grid[size - 1, 0] = r * scale
    state, r = next_rand(state)
    grid[size - 1, size - 1] = r * scale

    step = size - 1
    current = scale

    while step > 1:
        half = step // 2

        # =========================
        # Square step
        # =========================
        for x in range(half, size - 1, step):
            for y in range(half, size - 1, step):

                avg = (
                    grid[x - half, y - half] +
                    grid[x - half, y + half] +
                    grid[x + half, y - half] +
                    grid[x + half, y + half]
                ) * 0.25

                state, r = next_rand(state)
                grid[x, y] = avg + (r - 0.5) * current

        # =========================
        # Diamond step
        # =========================
        for x in range(0, size, half):

            # alternating offset
            if ((x // half) % 2) == 0:
                start_y = half
            else:
                start_y = 0

            for y in range(start_y, size, step):

                total = 0.0
                count = 0

                if x - half >= 0:
                    total += grid[x - half, y]
                    count += 1

                if x + half < size:
                    total += grid[x + half, y]
                    count += 1

                if y - half >= 0:
                    total += grid[x, y - half]
                    count += 1

                if y + half < size:
                    total += grid[x, y + half]
                    count += 1

                state, r = next_rand(state)
                grid[x, y] = (
                    total / count +
                    (r - 0.5) * current
                )

        step //= 2
        current *= roughness

    return grid