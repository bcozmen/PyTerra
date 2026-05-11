from numba import njit, prange
import numpy as np

@njit
def hash2d(x, y, seed):
    n = x * 374761393 + y * 668265263 + seed * 1446647
    n = (n ^ (n >> 13)) * 1274126177
    return ((n ^ (n >> 16)) & 0xffffffff) / 0xffffffff

@njit
def fade(t):
    return t * t * (3.0 - 2.0 * t)

@njit
def value_noise(x, y, seed):
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    x1 = x0 + 1
    y1 = y0 + 1

    sx = fade(x - x0)
    sy = fade(y - y0)

    n00 = hash2d(x0, y0, seed)
    n10 = hash2d(x1, y0, seed)
    n01 = hash2d(x0, y1, seed)
    n11 = hash2d(x1, y1, seed)

    ix0 = n00 * (1 - sx) + n10 * sx
    ix1 = n01 * (1 - sx) + n11 * sx

    return ix0 * (1 - sy) + ix1 * sy


@njit(parallel=True)
def fbm(X, Y, seed, octaves, persistence, lacunarity, base_freq):
    out = np.zeros_like(X)

    rows, cols = X.shape

    for i in prange(rows):
        for j in range(cols):

            amplitude = 1.0
            frequency = base_freq
            total = 0.0
            norm = 0.0

            for o in range(octaves):
                nx = X[i, j] * frequency
                ny = Y[i, j] * frequency

                val = value_noise(nx, ny, seed + o * 17)

                total += val * amplitude
                norm += amplitude

                amplitude *= persistence
                frequency *= lacunarity

            out[i, j] = total / norm

    return out

@njit(parallel=True)
def domain_warp( X, Y, seed, warp_strength, octaves, persistence, lacunarity, base_freq):
    wx = fbm(
        X,
        Y,
        seed + 101,
        octaves,
        persistence,
        lacunarity,
        base_freq
    )

    wy = fbm(
        X + 31.7,
        Y + 47.2,
        seed + 211,
        octaves,
        persistence,
        lacunarity,
        base_freq
    )

    return X + warp_strength * wx, Y + warp_strength * wy


# ---------------------------------------------------------------------------
# Ridge FBM  —  produces sharp mountain ridges instead of blobby hills
# ---------------------------------------------------------------------------

@njit(parallel=True)
def ridge_fbm(X, Y, seed, octaves, persistence, lacunarity, base_freq):
    """Ridged multi-fractal noise.

    Each octave contributes  ``1 - |2*noise - 1|``  (a "tent" function),
    which turns smooth hills into sharp ridgelines.  Successive octaves
    are weighted by the previous octave's value so ridges self-sharpen.

    Returns values roughly in [0, 1].
    """
    out = np.zeros_like(X)
    rows, cols = X.shape

    for i in prange(rows):
        for j in range(cols):
            amplitude = 1.0
            frequency = base_freq
            total = 0.0
            norm = 0.0
            prev = 1.0

            for o in range(octaves):
                nx = X[i, j] * frequency
                ny = Y[i, j] * frequency

                # Raw value in [0, 1] → ridge value in [0, 1]
                raw = value_noise(nx, ny, seed + o * 17)
                val = 1.0 - abs(raw * 2.0 - 1.0)
                val = val * val * prev   # sharpening feedback

                prev = val
                total += val * amplitude
                norm += amplitude

                amplitude *= persistence
                frequency *= lacunarity

            out[i, j] = total / norm

    return out


# ---------------------------------------------------------------------------
# Anisotropic domain warp  —  directional mountain ranges / elongated ridges
# ---------------------------------------------------------------------------

@njit(parallel=True)
def domain_warp_aniso(
    X, Y, seed,
    warp_strength_x, warp_strength_y,
    octaves, persistence, lacunarity, base_freq,
):
    """Domain warp with independent per-axis strength.

    Using ``warp_strength_x != warp_strength_y`` stretches noise features
    along one axis, producing elongated ridges / valleys typical of real
    mountain ranges instead of the circular blobs that symmetric warp
    creates.

    Parameters
    ----------
    warp_strength_x : displacement amount along the X axis.
    warp_strength_y : displacement amount along the Y axis.

    Returns
    -------
    (warped_X, warped_Y)
    """
    wx = fbm(X,        Y,        seed + 101, octaves, persistence, lacunarity, base_freq)
    wy = fbm(X + 31.7, Y + 47.2, seed + 211, octaves, persistence, lacunarity, base_freq)

    return X + warp_strength_x * wx, Y + warp_strength_y * wy
