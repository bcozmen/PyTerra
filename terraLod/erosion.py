import numpy as np
from numba import njit, prange

def hydraulic_erosion(grid, iterations, erosion_rate, deposition_rate, evaporation,
                      min_slope, inertia, gravity, capacity_factor,
                      max_steps, seed, **kwargs):
    return hydraulic_erosion_numba(grid, int(iterations), erosion_rate, deposition_rate, evaporation,
                                  min_slope, inertia, gravity, capacity_factor,
                                  max_steps, seed)

def thermal_erosion(grid, iterations, talus, **kwargs):
    return thermal_erosion_numba(grid, iterations, talus)

def air_erosion(grid, iterations, saltation_cells, wind_strength, threshold,
                erosion_rate, deposition_rate, avalanche_talus, wind_x, wind_y,
                **kwargs):
    return air_erosion_numba(grid, int(iterations), wind_x, wind_y,
                             wind_strength, threshold, int(saltation_cells),
                             erosion_rate, deposition_rate, avalanche_talus)


@njit(cache=True, parallel=True)
def hydraulic_erosion_numba(
    grid,
    iterations: int = 0,
    erosion_rate: float = 0.04,
    deposition_rate: float = 0.02,
    evaporation: float = 0.012,
    min_slope: float = 0.0005,
    inertia: float = 0.4,
    gravity: float = 10.0,
    capacity_factor: float = 12.0,
    max_steps: int = 600,
    seed: int = 0,
):
    h, w = grid.shape
    # Each parallel droplet reads from `out` (the live, accumulating map) so
    # that erosion by one droplet is visible to others.  Concurrent writes are
    # racy in theory, but the statistical result is a smooth blur that is
    # physically reasonable and prevents the blow-up caused by a frozen snapshot
    # where every droplet independently over-erodes the same source cells.
    out  = grid.copy()

    for i in prange(iterations):
        # Each droplet gets its own independent RNG state derived from seed + index.
        state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2654435761)
        state, _ = _lcg(state)   # warm up

        # Random start position (at least 1 cell away from border)
        state, rx = _lcg(state)
        state, ry = _lcg(state)
        px = 1.0 + rx * (h - 3)
        py = 1.0 + ry * (w - 3)

        water = 1.0
        sediment = 0.0
        vx = 0.0
        vy = 0.0
        speed = 0.0

        for _step in range(max_steps):
            ix = int(px)
            iy = int(py)

            if ix < 1 or ix >= h - 1 or iy < 1 or iy >= w - 1:
                # Clamp to valid range before depositing (ix/iy can reach
                # h-1 / w-1 on the boundary check, which are valid indices,
                # but a large velocity step could push them one cell further).
                cix = max(1, min(ix, h - 2))
                ciy = max(1, min(iy, w - 2))
                out[cix, ciy] += sediment
                break

            fx = px - ix
            fy = py - iy

            # Bilinear sample from the live map.  Concurrent reads are racy
            # in prange but produce at worst a slight smoothing effect, which
            # is far preferable to the blow-up caused by a frozen snapshot.
            h00 = out[ix,     iy    ]
            h10 = out[ix + 1, iy    ]
            h01 = out[ix,     iy + 1]
            h11 = out[ix + 1, iy + 1]

            # Bilinear height at current position
            cur_h = (
                h00 * (1.0 - fx) * (1.0 - fy)
                + h10 * fx       * (1.0 - fy)
                + h01 * (1.0 - fx) * fy
                + h11 * fx       * fy
            )

            # Gradient (descent direction)
            gx = (h10 - h00) * (1.0 - fy) + (h11 - h01) * fy
            gy = (h01 - h00) * (1.0 - fx) + (h11 - h10) * fx

            # Update velocity with inertia — steeper slopes accelerate more
            vx = vx * inertia - gx * (1.0 - inertia)
            vy = vy * inertia - gy * (1.0 - inertia)

            speed = (vx * vx + vy * vy) ** 0.5
            if speed < 1e-7:
                # Stalled droplet: deposit everything and stop
                out[ix,     iy    ] += sediment * (1.0 - fx) * (1.0 - fy)
                out[ix + 1, iy    ] += sediment * fx         * (1.0 - fy)
                out[ix,     iy + 1] += sediment * (1.0 - fx) * fy
                out[ix + 1, iy + 1] += sediment * fx         * fy
                break

            # Soft speed cap: allow physics-driven speed variation but
            # prevent runaway on steep cliffs.  Keeps erosion proportional
            # to actual slope rather than collapsing all droplets to speed=1.
            if speed > 4.0:
                inv = 4.0 / speed
                vx *= inv
                vy *= inv
                speed = 4.0

            # Step to next position
            nx = px + vx / speed   # unit-direction step so we always advance 1 cell
            ny = py + vy / speed
            nix = int(nx)
            niy = int(ny)
            if nix < 1 or nix >= h - 1 or niy < 1 or niy >= w - 1:
                out[ix, iy] += sediment
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
            # Capacity scales with slope, speed AND gravity — physically motivated
            capacity = max(slope, min_slope) * speed * water * gravity * capacity_factor

            if sediment > capacity:
                # Deposit excess sediment bilinearly
                deposit = deposition_rate * (sediment - capacity)
                sediment -= deposit
                out[ix,     iy    ] += deposit * (1.0 - fx) * (1.0 - fy)
                out[ix + 1, iy    ] += deposit * fx         * (1.0 - fy)
                out[ix,     iy + 1] += deposit * (1.0 - fx) * fy
                out[ix + 1, iy + 1] += deposit * fx         * fy
            else:
                # Erode bilinearly
                erode = erosion_rate * (capacity - sediment)
                sediment += erode
                out[ix,     iy    ] -= erode * (1.0 - fx) * (1.0 - fy)
                out[ix + 1, iy    ] -= erode * fx         * (1.0 - fy)
                out[ix,     iy + 1] -= erode * (1.0 - fx) * fy
                out[ix + 1, iy + 1] -= erode * fx         * fy

            water *= 1.0 - evaporation
            if water < 0.01:
                # Evaporated: deposit remaining sediment at current position
                out[ix,     iy    ] += sediment * (1.0 - fx) * (1.0 - fy)
                out[ix + 1, iy    ] += sediment * fx         * (1.0 - fy)
                out[ix,     iy + 1] += sediment * (1.0 - fx) * fy
                out[ix + 1, iy + 1] += sediment * fx         * fy
                break

            px = nx
            py = ny

        else:
            # Droplet exhausted max_steps without any break — deposit remaining
            # sediment so mass is conserved.
            ix = int(px)
            iy = int(py)
            cix = max(1, min(ix, h - 2))
            ciy = max(1, min(iy, w - 2))
            out[cix, ciy] += sediment

    return out


# ---------------------------------------------------------------------------
# Thermal erosion
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def thermal_erosion_numba(grid, iterations: int = 0, talus: float = 0.025):
    """Talus-angle thermal erosion — parallel 9-colour implementation.

    Redistributes material from steep slopes to their lower neighbours
    when the height difference exceeds *talus*.

    Parallelism strategy — 9-colour (3×3 tile) decomposition
    ---------------------------------------------------------
    Each iteration is split into 9 sequential sub-passes, one per colour
    ``(ci, cj)`` where ``ci = i % 3`` and ``cj = j % 3``.  Within a
    sub-pass all cells of the same colour are processed in parallel via
    ``prange``.

    Correctness proof: two cells of the same colour are at least 3 cells
    apart in every dimension.  Their 8-neighbourhoods extend only 1 cell
    in each direction, so neighbourhoods of same-colour cells never overlap.
    Each cell and its neighbours are therefore touched by exactly one thread
    per sub-pass → **zero race conditions**, no locking required.

    Other improvements
    ------------------
    * Diagonal neighbours use ``talus × √2`` (distance-corrected).
    * Material is distributed proportionally to each neighbour's excess,
      producing smooth alluvial fans.
    * Mass is exactly conserved.

    Parameters
    ----------
    grid       : 2-D float32 height map.
    iterations : number of full passes (each pass = 9 parallel sub-passes).
    talus      : maximum stable height difference for axial neighbours.

    Returns
    -------
    Modified copy of *grid*.
    """
    h, w = grid.shape
    out = grid.copy()
    SQRT2 = 1.41421356

    for _ in range(iterations):
        # 9 sub-passes — one per (ci, cj) colour in {0,1,2}²
        for ci in range(3):
            for cj in range(3):
                # Number of cells of this colour (ceiling division)
                ni = (h - 2 - ci + 2) // 3   # rows in [1, h-2] with row%3 == ci
                nj = (w - 2 - cj + 2) // 3

                for pi in prange(ni):
                    i = 1 + ci + pi * 3
                    if i >= h - 1:
                        continue
                    for pj in range(nj):
                        j = 1 + cj + pj * 3
                        if j >= w - 1:
                            continue

                        total_excess = 0.0

                        for di in range(-1, 2):
                            for dj in range(-1, 2):
                                if di == 0 and dj == 0:
                                    continue
                                t = talus * SQRT2 if (di != 0 and dj != 0) else talus
                                diff = out[i, j] - out[i + di, j + dj]
                                if diff > t:
                                    total_excess += (diff - t)

                        if total_excess > 0.0:
                            for di in range(-1, 2):
                                for dj in range(-1, 2):
                                    if di == 0 and dj == 0:
                                        continue
                                    t = talus * SQRT2 if (di != 0 and dj != 0) else talus
                                    diff = out[i, j] - out[i + di, j + dj]
                                    if diff > t:
                                        excess = diff - t
                                        # Distribute proportionally: each neighbour receives
                                        # (excess / total_excess) of the total material to move
                                        # (total to move = total_excess * 0.5 for stability).
                                        # Simplifies to: move = excess * 0.5
                                        move = 0.5 * excess
                                        out[i + di, j + dj] += move
                                        out[i, j]           -= move

    return out

@njit(cache=True, inline='always')
def _lcg(s):
    """64-bit LCG step.  Returns (new_state, uniform float in [0, 1))."""
    s = (
        s * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
    ) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return s, float(s >> np.uint64(11)) / float(np.uint64(1) << np.uint64(53))



@njit(cache=True, parallel=True)
def air_erosion_numba(
    grid,
    iterations=8,
    wind_x=1.0,
    wind_y=0.3,
    wind_strength=0.06,
    threshold=0.008,
    saltation_cells=4,
    erosion_rate=0.012,
    deposition_rate=1.0,
    avalanche_talus=0.05,
):
    """Aeolian erosion via saltation + slip-face avalanching.

    Physics
    -------
    Shear stress on a cell is proportional to its *windward slope* — how much
    it rises above its upwind neighbour.  Windward faces (positive slope) are
    eroded; lee faces have zero windward slope so they are naturally protected
    without a separate shadow parameter.

    Eroded grains hop exactly *saltation_cells* downwind and are deposited
    there (``deposition_rate = 1.0`` → mass exactly conserved).

    A slip-face avalanche pass after each saltation sweep enforces the maximum
    stable slope *avalanche_talus*, which causes material to pile up into
    dune-like asymmetric ridges.

    Parallelism
    -----------
    Saltation pass — sweep the *wind axis* sequentially (upwind → downwind)
    so that each row's deposits land in a row that has not yet been processed,
    eliminating all RAW hazards.  The *perpendicular* axis is independent
    within a row and runs in parallel via ``prange``.

    * ``di != 0``: outer loop over rows (sequential), inner over columns
      (``prange``).  Deposits from row *i* land in row *i + di* which is
      processed later → race-free.
    * ``di == 0`` (pure crosswind): roles swap — outer loop over columns
      (sequential), inner over rows (``prange``).

    Avalanche pass — 9-colour (3×3 tile) decomposition identical to the one
    used in ``thermal_erosion_numba``: same-colour cells are ≥ 3 apart in
    every dimension so their 8-neighbourhoods never overlap → zero races.

    Parameters
    ----------
    grid             : normalised [0, 1] height map.
    iterations       : number of full wind passes (each = saltation + avalanche).
    wind_x / wind_y  : wind direction vector (need not be unit length).
    wind_strength    : shear coefficient — multiplied by windward slope.
    threshold        : minimum shear required to mobilise grains.
    saltation_cells  : hop length in grid cells (derive from physical metres
                       via ``saltation_base_m / dx_m`` before calling).
    erosion_rate     : fraction of available excess shear converted to volume.
    deposition_rate  : fraction of eroded volume deposited at landing
                       (1.0 → exact mass conservation).
    avalanche_talus  : maximum stable slope in normalised height units;
                       derive from angle via ``tan(θ) × dx_m / dz_m``.
    """
    h, w = grid.shape
    out = grid.copy()
    SQRT2 = 1.41421356

    # Normalise wind direction
    wl = (wind_x * wind_x + wind_y * wind_y) ** 0.5 + 1e-12
    wx = wind_x / wl
    wy = wind_y / wl

    # Integer saltation hop (downwind)
    di = int(round(wx * saltation_cells))
    dj = int(round(wy * saltation_cells))
    if di == 0 and dj == 0:
        di = 1  # guarantee at least 1-cell transport

    # Upwind look-back offset (one hop upstream, for windward slope)
    ui = -di
    uj = -dj

    for _ in range(iterations):

        # ---------------------------------------------------------------
        # Saltation pass
        # Sequential axis: wind direction  |  Parallel axis: perpendicular
        # ---------------------------------------------------------------
        if di != 0:
            # Sweep rows upwind → downwind; columns in parallel.
            # Deposits from (i, j) land at (i+di, j+dj) — a different row
            # that has not been processed yet → no RAW hazard across threads.
            for ii in range(h - 2):
                i = (1 + ii) if wx >= 0.0 else (h - 2 - ii)
                for j in prange(1, w - 1):
                    z = out[i, j]
                    pui = i + ui
                    puj = j + uj
                    if 0 <= pui < h and 0 <= puj < w:
                        upwind_z = out[pui, puj]
                    else:
                        upwind_z = z  # boundary → treat as flat
                    windward_slope = z - upwind_z
                    if windward_slope <= 0.0:
                        continue
                    shear = wind_strength * windward_slope
                    if shear < threshold:
                        continue
                    erode = erosion_rate * (shear - threshold)
                    if erode > out[i, j]:
                        erode = out[i, j]
                    ni = i + di
                    nj = j + dj
                    if 1 <= ni < h - 1 and 1 <= nj < w - 1:
                        out[i, j]   -= erode
                        out[ni, nj] += erode * deposition_rate
        else:
            # di == 0 (pure crosswind): sweep columns upwind → downwind;
            # rows in parallel.  Deposits land at (i, j+dj) — a different
            # column → no RAW hazard across threads.
            for jj in range(w - 2):
                j = (1 + jj) if wy >= 0.0 else (w - 2 - jj)
                for i in prange(1, h - 1):
                    z = out[i, j]
                    pui = i + ui
                    puj = j + uj
                    if 0 <= pui < h and 0 <= puj < w:
                        upwind_z = out[pui, puj]
                    else:
                        upwind_z = z  # boundary → treat as flat
                    windward_slope = z - upwind_z
                    if windward_slope <= 0.0:
                        continue
                    shear = wind_strength * windward_slope
                    if shear < threshold:
                        continue
                    erode = erosion_rate * (shear - threshold)
                    if erode > out[i, j]:
                        erode = out[i, j]
                    ni = i + di
                    nj = j + dj
                    if 1 <= ni < h - 1 and 1 <= nj < w - 1:
                        out[i, j]   -= erode
                        out[ni, nj] += erode * deposition_rate

        # ---------------------------------------------------------------
        # Slip-face avalanche pass — 9-colour parallel decomposition
        # Same-colour cells are ≥ 3 apart in every dimension so their
        # 8-neighbourhoods never overlap → zero race conditions.
        # ---------------------------------------------------------------
        for ci in range(3):
            for cj in range(3):
                ni_count = (h - 2 - ci + 2) // 3
                nj_count = (w - 2 - cj + 2) // 3
                for pi in prange(ni_count):
                    i = 1 + ci + pi * 3
                    if i >= h - 1:
                        continue
                    for pj in range(nj_count):
                        j = 1 + cj + pj * 3
                        if j >= w - 1:
                            continue
                        for ddi in range(-1, 2):
                            for ddj in range(-1, 2):
                                if ddi == 0 and ddj == 0:
                                    continue
                                t = avalanche_talus * SQRT2 if (ddi != 0 and ddj != 0) else avalanche_talus
                                diff = out[i, j] - out[i + ddi, j + ddj]
                                if diff > t:
                                    move = (diff - t) * 0.5
                                    out[i, j]              -= move
                                    out[i + ddi, j + ddj]  += move

    return out