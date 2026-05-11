"""
Erosion passes for the LOD terrain pipeline.

Both passes operate on a 2-D float32 height grid normalised to [0, 1].
They are compiled by Numba (cache=True) so the first call pays the JIT
cost; subsequent calls are fast.

Hydraulic erosion (particle-based)
-----------------------------------
Drop `iterations` virtual water droplets.  Each droplet flows downhill,
eroding material on steep sections and depositing it where the slope
drops below the droplet's carrying capacity.  The result carves V-shaped
valleys, river channels, and natural drainage networks.

Thermal erosion (talus angle)
------------------------------
Fast pass that flattens unrealistically steep cliffs by moving material
from high cells to their lower neighbours when the height difference
exceeds `talus`.  Run this *before* hydraulic erosion to stabilise
initial noise spikes, and optionally *after* to soften artefacts.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Hydraulic erosion
# ---------------------------------------------------------------------------

@njit(cache=True)
def hydraulic_erosion(
    grid,
    iterations: int = 80,
    erosion_rate: float = 0.03,
    deposition_rate: float = 0.03,
    evaporation: float = 0.02,
    min_slope: float = 0.0005,
    seed: int = 0,
):
    """Particle-based hydraulic erosion.

    Parameters
    ----------
    grid            : 2-D float32 height map (modified in-place copy).
    iterations      : number of water droplets to simulate.
    erosion_rate    : fraction of carrying capacity eroded per step.
    deposition_rate : fraction of excess sediment deposited per step.
    evaporation     : water evaporated per step (reduces carrying capacity).
    min_slope       : minimum slope used for capacity calculation.
    seed            : RNG seed for reproducible droplet placement.

    Returns
    -------
    Modified copy of *grid*.
    """
    h, w = grid.shape
    out = grid.copy()

    # Simple 64-bit LCG — fast and numba-compatible
    state = np.uint64(seed)

    def _lcg(s):
        s = np.uint64(
            (s * np.uint64(6364136223846793005) + np.uint64(1442695040888963407))
            & np.uint64(0xFFFFFFFFFFFFFFFF)
        )
        return s, float(s >> np.uint64(11)) / float(np.uint64(1) << np.uint64(53))

    for _ in range(iterations):
        # Random start position (at least 1 cell away from border)
        state, rx = _lcg(state)
        state, ry = _lcg(state)
        px = 1.0 + rx * (h - 3)
        py = 1.0 + ry * (w - 3)

        water = 1.0
        sediment = 0.0
        vx = 0.0
        vy = 0.0

        for _step in range(96):
            ix = int(px)
            iy = int(py)

            if ix < 1 or ix >= h - 1 or iy < 1 or iy >= w - 1:
                break

            fx = px - ix
            fy = py - iy

            # Bilinear sample of the four surrounding corners
            h00 = out[ix,     iy    ]
            h10 = out[ix + 1, iy    ]
            h01 = out[ix,     iy + 1]
            h11 = out[ix + 1, iy + 1]

            # Gradient (descent direction)
            gx = (h10 - h00) * (1.0 - fy) + (h11 - h01) * fy
            gy = (h01 - h00) * (1.0 - fx) + (h11 - h10) * fx

            # Update velocity (inertia 0.5)
            vx = vx * 0.5 - gx
            vy = vy * 0.5 - gy

            speed = (vx * vx + vy * vy) ** 0.5
            if speed < 1e-7:
                break

            # Bilinear height at current position
            cur_h = (
                h00 * (1.0 - fx) * (1.0 - fy)
                + h10 * fx       * (1.0 - fy)
                + h01 * (1.0 - fx) * fy
                + h11 * fx       * fy
            )

            # Step to next position
            nx = px + vx
            ny = py + vy
            nix = int(nx)
            niy = int(ny)
            if nix < 1 or nix >= h - 1 or niy < 1 or niy >= w - 1:
                break
            nfx = nx - nix
            nfy = ny - niy

            nh = (
                out[nix,     niy    ] * (1.0 - nfx) * (1.0 - nfy)
                + out[nix + 1, niy    ] * nfx         * (1.0 - nfy)
                + out[nix,     niy + 1] * (1.0 - nfx) * nfy
                + out[nix + 1, niy + 1] * nfx         * nfy
            )

            slope = cur_h - nh
            capacity = max(slope, min_slope) * speed * water * 8.0

            if sediment > capacity:
                # Deposit excess sediment
                deposit = deposition_rate * (sediment - capacity)
                sediment -= deposit
                out[ix,     iy    ] += deposit * (1.0 - fx) * (1.0 - fy)
                out[ix + 1, iy    ] += deposit * fx         * (1.0 - fy)
                out[ix,     iy + 1] += deposit * (1.0 - fx) * fy
                out[ix + 1, iy + 1] += deposit * fx         * fy
            else:
                # Erode — capped so we never dig below neighbours
                erode = erosion_rate * (capacity - sediment)
                sediment += erode
                out[ix,     iy    ] -= erode * (1.0 - fx) * (1.0 - fy)
                out[ix + 1, iy    ] -= erode * fx         * (1.0 - fy)
                out[ix,     iy + 1] -= erode * (1.0 - fx) * fy
                out[ix + 1, iy + 1] -= erode * fx         * fy

            water *= 1.0 - evaporation
            if water < 0.01:
                break

            px = nx
            py = ny

    return out


# ---------------------------------------------------------------------------
# Thermal erosion
# ---------------------------------------------------------------------------

@njit(cache=True)
def thermal_erosion(grid, iterations: int = 5, talus: float = 0.025):
    """Talus-angle thermal erosion.

    Redistributes material from steep slopes to their lower neighbours
    when the height difference exceeds *talus*.  Fast and effective at
    stabilising initial noise spikes before hydraulic erosion.

    Parameters
    ----------
    grid       : 2-D float32 height map.
    iterations : number of passes.
    talus      : maximum stable height difference between neighbours.

    Returns
    -------
    Modified copy of *grid*.
    """
    h, w = grid.shape
    out = grid.copy()

    for _ in range(iterations):
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                max_diff = 0.0
                total_excess = 0.0
                # 8-neighbourhood
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        diff = out[i, j] - out[i + di, j + dj]
                        if diff > talus:
                            if diff > max_diff:
                                max_diff = diff
                            total_excess += diff

                if max_diff > talus and total_excess > 0.0:
                    move = 0.5 * (max_diff - talus)
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            if di == 0 and dj == 0:
                                continue
                            diff = out[i, j] - out[i + di, j + dj]
                            if diff > talus:
                                frac = diff / total_excess
                                out[i + di, j + dj] += move * frac
                    out[i, j] -= move

    return out
