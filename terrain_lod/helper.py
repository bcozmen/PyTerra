import numpy as np
from numba import njit
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import time 
from functools import wraps, lru_cache



def time_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


@lru_cache(maxsize=64)
def _get_mask(h, w, cutoff, rolloff):
    """
    Precompute and cache frequency-domain low-pass mask.
    """
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)

    return 1.0 / (1.0 + (radius / cutoff) ** rolloff)


def fft_lowpass(height, cutoff=0.15, rolloff=3.0):
    """
    Low-pass filter using FFT with edge-safe padding.

    The previous implementation applied the FFT directly to the input which
    causes circular convolution and can produce spurious values at the
    boundaries. To avoid that we reflect-pad the input, perform the FFT in
    the larger domain, apply the frequency mask sized for the padded shape,
    then inverse-FFT and crop the result back to the original shape.

    The padding size is chosen as half the longest dimension which is a good
    trade-off between reducing wrap-around and keeping the frequency
    resolution reasonable.
    """
    h, w = height.shape

    # Choose padding: half the longest dimension (can be tuned).
    pad = max(1, max(h, w) // 2)

    # Reflect-pad to minimise seams (mirror avoids introducing DC offsets)
    padded = np.pad(height, ((pad, pad), (pad, pad)), mode='reflect')
    ph, pw = padded.shape

    # FFT
    F = np.fft.fft2(padded)

    # Cached mask lookup for padded shape
    mask = _get_mask(ph, pw, cutoff, rolloff)

    # Apply filter
    F *= mask

    # Inverse FFT and crop back to original size
    filtered = np.fft.ifft2(F).real
    result = filtered[pad:pad + h, pad:pad + w]
    return result

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

