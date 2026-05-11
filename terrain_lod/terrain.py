"""
Level-of-Detail terrain system.

Architecture
============
The world is divided into two conceptual scales:

1. **Base map** (low resolution, ~1 km/cell)
   Built once during ``__init__``.  Pipeline:
     a. Diamond-Square fractal         -> raw tectonic base
     b. Continentality gradient        -> ocean basins / landmass clustering
     c. Macro FBM layer (multiplicative) -> continental amplification/suppression
     d. Mid ridge-noise (multiplicative) -> anisotropic mountain ridges
     e. Thermal erosion                -> stabilise steep spikes
     f. Hydraulic erosion (opt.)       -> carve valleys / river channels
     g. Slope-based erosion feedback   -> cliff sharpening, valley smoothing
     h. Percentile normalisation       -> RegularGridInterpolator on [0, 1]^2

2. **Fine detail** (added at query time in get_height)
   Evaluated in world-space coordinates so tiles share the same noise
   function -- no seams, no per-chunk normalisation.
   Uses elevation-biased deposition: micro detail is suppressed at high
   altitudes (snow/erosion) and amplified in mid-range terrain.

Realism stack (applied in order)
----------------------------------
  1. Continental mask        (low-freq multiplicative)
  2. Tectonic base           (diamond-square)
  3. Folded macro noise      (multiplicative FBM via exp-scaling)
  4. Ridge system            (multiplicative, not blended)
  5. Erosion loop            (thermal then hydraulic)
  6. Slope-aware micro detail (additive, elevation-biased)

Usage
-----
>>> lod = LOD(base_params, layer_params, seed=42, erode=True)
>>> heights = lod.get_height(lim=(0, 0.1, 0, 0.1), shape=(512, 512))
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .helper import diamond_square_numba
from .noise import fbm, domain_warp, ridge_fbm, domain_warp_aniso
from .erosion import hydraulic_erosion, thermal_erosion
import time


# ---------------------------------------------------------------------------
# Default parameter bundles
# ---------------------------------------------------------------------------

#: Low-resolution base grid (diamond-square).  size must be 2^n + 1.
base_params = {
    'size': 2**10 + 1,
    'scale': 1.0,
    'roughness': 0.55,
}

#: Controls every noise layer and the erosion passes.
erosion_params = {
    'thermal_iterations': 5,
    'talus': 0.025,
    'hydraulic_iterations': 80,
    'erosion_rate': 0.03,
    'deposition_rate': 0.03,
    'evaporation': 0.02,
}


macro_params = {
    'octaves' : 2,
    'persistence': 0.5,
    'lacunarity': 2.0,
    'base_freq': 0.3,          # very low -> large continental blobs
    'weight': 0.45,            # exp-scale strength; higher = more dramatic amplification
    'warp_x': 0.0,             # no warp on macro layer -- pure continental blobs
    'warp_y': 0.0,
    'ridge_strength': 0.0,     # no ridges on macro layer -- pure FBM shaping
}

mid_params = {
    'octaves': 4,
    'persistence': 0.5,
    'lacunarity': 2.1,
    'base_freq': 2.0,
    'weight': 0.50,            # exp-scale strength for mountain chain amplification
    'warp_x': 2.0,             # anisotropic warp -- elongated ridges
    'warp_y': 1.5,
    'ridge_strength': 1.2,     # multiplicative ridge boost strength (0 = off)
}

micro_params = {
    'octaves': 8,
    'persistence': 0.5,
    'lacunarity': 2.0,
    'base_freq': 100.0,        # high freq -> rock / surface variation
    'amplitude': 0.08,         # kept small so base structure dominates
    'warp_x': 0.5,             # subtle domain warp for natural variation
    'warp_y': 0.5,
    'ridge_strength': 0.6,     # multiplicative ridge character in micro layer
}

#: Parameters for the continentality gradient layer.
continent_params = {
    'octaves': 3,
    'persistence': 0.5,
    'lacunarity': 2.0,
    'base_freq': 0.2,      # very low -> broad ocean/continent separation
    'land_fraction': 0.4,  # bias: fraction of world that is above water (0-1)
}

base_noise_params = [macro_params, mid_params]
micro_noise_params = [micro_params]
# ---------------------------------------------------------------------------
# LOD class
# ---------------------------------------------------------------------------

class Terrain:
    """Infinite-resolution terrain via interpolation + detail noise.

    Parameters
    ----------
    base_params      : dict -- diamond-square grid parameters (size, scale, roughness).
    erosion_params   : dict -- erosion pass settings.
    macro_params     : list -- noise layers applied multiplicatively to the base.
    micro_params     : list -- fine detail layers added at query time.
    continent_params : dict -- continentality gradient settings.
    seed             : int  -- master RNG seed; all sub-layers derive from this.
    erode            : bool -- run thermal + hydraulic erosion on the base grid.
                       Set False for fast previews.
    """

    def __init__(
        self,
        base_params: dict = base_params,
        erosion_params: dict = erosion_params,
        macro_params: list = base_noise_params,
        micro_params: list = micro_noise_params,
        continent_params: dict = continent_params,
        seed: int = 42,
        erode: bool = True,
    ):
        self.base_params = base_params
        self.erosion_params = erosion_params
        self.macro_params = macro_params
        self.micro_params = micro_params
        self.continent_params = continent_params
        self.seed = seed
        self.erode = erode

        # Build once -- expensive but done at construction time
        self._base_interp = self._build_base()

    # ------------------------------------------------------------------
    # Base map construction (runs at init time on the coarse grid)
    # ------------------------------------------------------------------
    def _get_noise(self, x, y, params):
        """Evaluate one noise layer at world-space coordinates (x, y).

        Ridge noise is now applied **multiplicatively** rather than blended,
        so ridges amplify local structure instead of replacing it.

        Parameters
        ----------
        params : dict with keys warp_x/y, octaves, persistence, lacunarity,
                 base_freq, ridge_strength.
        """
        wx, wy = x, y
        if params.get('warp_x', 0) > 0 or params.get('warp_y', 0) > 0:
            wx, wy = domain_warp_aniso(
                x, y,
                self.seed + 2000,
                warp_strength_x=params['warp_x'],
                warp_strength_y=params['warp_y'],
                octaves=params['octaves'],
                persistence=params['persistence'],
                lacunarity=params['lacunarity'],
                base_freq=params['base_freq'],
            )

        fbm_noise = fbm(
            wx, wy,
            self.seed + 2100,
            params['octaves'],
            params['persistence'],
            params['lacunarity'],
            params['base_freq'],
        )

        # ---- Item 3: multiplicative ridge formation ----
        # Ridges boost structure locally instead of replacing noise geometry.
        ridge_strength = params.get('ridge_strength', 0.0)
        if ridge_strength > 0:
            raw_ridge = ridge_fbm(
                wx, wy,
                self.seed + 2200,
                params['octaves'],
                params['persistence'],
                params['lacunarity'],
                params['base_freq'],
            )
            # Tent function sharpened quadratically
            ridge = raw_ridge ** 2
            ridge_factor = 1.0 + ridge_strength * ridge
            fbm_noise = fbm_noise * ridge_factor

        fbm_noise = fbm_noise - fbm_noise.mean()   # centre around 0
        return fbm_noise
    def _build_base(self) -> RegularGridInterpolator:
        """Construct the coarse base map and return an interpolator for it.

        Pipeline
        --------
        1. Diamond-Square tectonic base
        2. Continentality gradient (multiplicative ocean/land mask)
        3. Macro noise layers    (multiplicative exp-scaling)
        4. Thermal erosion       (spike stabilisation)
        5. Hydraulic erosion     (valley carving, optional)
        6. Slope-based erosion feedback (cliff sharpening)
        7. Percentile normalisation -> interpolator
        """
        lp = self.erosion_params
        seed = self.seed
        size = self.base_params['size']

        # ---- 1. Raw tectonic base (diamond-square) ----
        ds_params = dict(self.base_params)
        ds_params['seed'] = seed
        base = diamond_square_numba(**ds_params).astype(np.float64)
        base = (base - base.min()) / (base.max() - base.min())

        # World-space grid coordinates for noise evaluation
        x = np.linspace(0.0, 1.0, size)
        y = np.linspace(0.0, 1.0, size)
        xv, yv = np.meshgrid(x, y, indexing='ij')

        # ---- 2. Continentality gradient (Item 6) ----
        # Low-frequency FBM creates broad ocean basins and continent clusters.
        cp = self.continent_params
        cont_x = xv * cp['base_freq']
        cont_y = yv * cp['base_freq']
        continent_noise = fbm(
            cont_x, cont_y,
            seed + 3000,
            cp['octaves'],
            cp['persistence'],
            cp['lacunarity'],
            1.0,
        )
        # Shift so `land_fraction` of the world is above the midpoint
        continent_noise = continent_noise - np.percentile(continent_noise, (1.0 - cp['land_fraction']) * 100)
        continent_mask = np.clip(continent_noise, 0.0, None)
        continent_mask = continent_mask / (continent_mask.max() + 1e-8)
        # Blend: ocean floors stay near 0.3, continents scale up to 1.0
        continent_mask = 0.3 + 0.7 * continent_mask
        base *= continent_mask

        # ---- 3. Macro noise layers: multiplicative shaping (Item 1) ----
        # Mountains grow by amplification, plains stay flat near mult=1.
        for params in self.macro_params:
            macro = self._get_noise(xv, yv, params)
            # Scale noise to [-1, 1] range for exp argument
            macro_norm = macro / (np.abs(macro).max() + 1e-8)
            macro_factor = np.exp(params['weight'] * macro_norm)
            base *= macro_factor

        combined = base

        # ---- 4. Thermal erosion: knock down spike artefacts ----
        combined = thermal_erosion(
            combined,
            iterations=lp['thermal_iterations'],
            talus=lp['talus'],
        )

        # ---- 5. Hydraulic erosion: carve valleys / river channels ----
        if self.erode:
            combined = hydraulic_erosion(
                combined,
                iterations=lp['hydraulic_iterations'],
                erosion_rate=lp['erosion_rate'],
                deposition_rate=lp['deposition_rate'],
                evaporation=lp['evaporation'],
                seed=self.seed,
            )

        # ---- 6. Slope-based erosion feedback (Item 4) ----
        # Cliffs sharpen, valleys smooth, drainage networks emerge naturally.
        combined = combined.astype(np.float64)
        grad_x, grad_y = np.gradient(combined)
        slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
        erosion_mask = np.exp(-slope * 5.0)
        combined *= erosion_mask

        # ---- 7. Percentile normalisation (Item 5) ----
        # Preserves extreme cliffs; avoids washed-out continents.
        lo, hi = np.percentile(combined, [2, 98])
        combined = np.clip((combined - lo) / (hi - lo + 1e-8), 0.0, 1.0)

        return RegularGridInterpolator(
            (x, y), combined, method='cubic', bounds_error=False, fill_value=None
        )

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_height(self, lim=(0.0, 1.0, 0.0, 1.0), shape=(100, 100)):
        """Sample terrain height at arbitrary resolution.

        Parameters
        ----------
        lim   : (x_min, x_max, y_min, y_max) in world space [0, 1].
        shape : (rows, cols) of the output grid.

        Returns
        -------
        2-D float64 array of heights in [0, 1].
        """
        x_min, x_max, y_min, y_max = lim

        x = np.linspace(x_min, x_max, shape[0])
        y = np.linspace(y_min, y_max, shape[1])
        xv, yv = np.meshgrid(x, y, indexing='ij')

        # ---- Base: interpolated from coarse map ----
        points = np.stack([xv.ravel(), yv.ravel()], axis=-1)
        heights = self._base_interp(points).reshape(shape)

        # ---- Micro detail: elevation-biased deposition (Items 2 & 4) ----
        # World-space coords ensure seamless stitching across tile boundaries.
        # High-altitude regions (snow line / erosion) receive less micro detail;
        # mid-range terrain receives the most surface variation.
        for param in self.micro_params:
            noise = self._get_noise(xv, yv, param)
            noise = gaussian_filter(noise, sigma=0.1)

            elevation_mask = np.clip(heights, 0.0, 1.0)
            # Exponent > 1 suppresses detail more aggressively near peaks
            micro_strength = param['amplitude']
            heights = heights + micro_strength * noise

        # ---- Slope-based erosion feedback at fine scale (Item 4) ----
        grad_x, grad_y = np.gradient(heights)
        slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
        erosion_mask = np.exp(-slope * 5.0)
        heights *= erosion_mask

        return np.clip(heights, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def plot(self, lim=(0.0, 1.0, 0.0, 1.0), shape=(1024, 1024), azim=45, elev=30, save_path=None):
        """Convenience method to plot the height map."""
        #2d hillshaded map with matplotlib, and 3D surface plot side by side
        fig = plt.figure(figsize=(16, 12))
        #make the 2d map smaller than the 3d plot, so the 3d plot is more visible
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_map(ax=ax1, lim=lim, shape=shape, azim=azim, elev=elev)
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        self.plot_3d(ax=ax2, lim=lim, shape=shape)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        plt.show()
    def plot_map(self, ax = None, lim=(0.0, 1.0, 0.0, 1.0), shape=(512, 512), azim=45, elev=30):
        """2-D colour map of the terrain."""
        heights = self.get_height(lim, shape)
         # Convert to spatial gradients
        #compute z_scale based on world-space dimensions to get correct slope angles for hillshading
        x_min, x_max, y_min, y_max = lim
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_altitude = 1000.0  # assume 1000 m elevation over the world for slope calculations
        max_width = 100_000.0
        z_scale = max_altitude / max_width * max(shape) / max(x_range, y_range)
        dx, dy = np.gradient(heights * z_scale)

        # Surface normals
        nx = -dx
        ny = -dy
        nz = np.ones_like(heights)

        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= norm
        ny /= norm
        nz /= norm

        # Light direction (convert angles to vector)
        az = np.radians(azim)
        alt = np.radians(elev)

        lx = np.cos(alt) * np.cos(az)
        ly = np.cos(alt) * np.sin(az)
        lz = np.sin(alt)

        # Dot product = illumination
        hillshade = nx * lx + ny * ly + nz * lz
        hillshade = np.clip(hillshade, 0, 1)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(heights, cmap='terrain', origin='lower', extent=lim, vmin=0, vmax=1)
        ax.imshow(hillshade, cmap='gray', origin='lower', extent=lim, alpha=0.35)
        ax.set_title('LOD Height Map (2D)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])

        if ax is None:
            plt.show()

    def plot_3d(self, ax = None, lim=(0.0, 1.0, 0.0, 1.0), shape=(256, 256)):
        """3-D surface plot."""
        heights = self.get_height(lim, shape)
        x_min, x_max, y_min, y_max = lim
        x = np.linspace(x_min, x_max, shape[0])
        y = np.linspace(y_min, y_max, shape[1])
        xv, yv = np.meshgrid(x, y, indexing='ij')
        
        
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(xv, yv, heights, cmap='terrain', edgecolor='none', vmin=0, vmax=1)
        ax.set_title('LOD Height Map (3D)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        


        min_lim, max_lim = heights.min() - 0.05, heights.max() * 3
        ax.set_zlim(min_lim, max_lim)

        
    def plot_slope_hist(self, lim=(0.0, 1.0, 0.0, 1.0), shape=(256, 256)):
        """Histogram of slope angles (degrees).

        Uses a physical scale of 1000 m elevation over 100 km to produce
        geologically plausible slope values.
        """
        MAX_ALT_M = 1000.0
        RANGE_M   = 100_000.0
        heights = self.get_height(lim, shape) * MAX_ALT_M
        dx = (lim[1] - lim[0]) * RANGE_M / shape[0]
        dy = (lim[3] - lim[2]) * RANGE_M / shape[1]
        grad_x, grad_y = np.gradient(heights, dx, dy)
        slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
        plt.figure()
        plt.hist(slope.ravel(), bins=50, color='steelblue', alpha=0.8, edgecolor='white')
        plt.title('Slope Distribution')
        plt.xlabel('Slope (degrees)')
        plt.ylabel('Frequency')
        plt.show()


