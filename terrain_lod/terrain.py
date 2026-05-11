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

from .helper import diamond_square_numba, fft_lowpass
from .noise import fbm, domain_warp, ridge_fbm, domain_warp_aniso
from .erosion import hydraulic_erosion, thermal_erosion
from .plotter import hillshade, plot2D, plot3D, plot

from .params import (
    world_params,
    base_params,
    erosion_params,
    continent_params,
    base_noise_params,
    micro_noise_params,
)

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
        world_params: dict = world_params,
        base_params: dict = base_params,
        erosion_params: dict = erosion_params,
        macro_params: list = base_noise_params,
        micro_params: list = micro_noise_params,
        continent_params: dict = continent_params,
        seed: int = 42,
        erode: bool = True,
    ):
        self.world_params = world_params
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
        #base *= continent_mask

        import matplotlib.pyplot as plt
        plt.imshow(base, cmap='terrain')
        plt.title('Base Map')
        plt.colorbar(label='Height')
        plt.show()

        # ---- 3. Macro noise layers: multiplicative shaping (Item 1) ----
        # Mountains grow by amplification, plains stay flat near mult=1.
        for params in self.macro_params:
            macro = self._get_noise(xv, yv, params)
            # Scale noise to [-1, 1] range for exp argument
            macro_norm = macro / (np.abs(macro).max() + 1e-8)
            
            #base *= macro_factor
            weight = params['weight'] / (params['base_freq']+1)  #scale weight by base_freq to keep the same physical influence regardless of the frequency
            base = base * (1 - weight) + macro_norm * weight  #blend rather than pure multiplication to preserve some of the base structure
            plt.imshow(base, cmap='terrain')
            plt.title('Macro Noise Map')
            plt.colorbar(label='Height')
            plt.show()


        combined = base

        # ---- 4. Thermal erosion: knock down spike artefacts ----
        combined = thermal_erosion(
            combined,
            iterations=lp['thermal_iterations'],
            talus=lp['talus'],
        )

        plt.imshow(combined, cmap='terrain')
        plt.title('Thermal Erosion Map')
        plt.colorbar(label='Height')
        plt.show()

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

        plt.imshow(combined, cmap='terrain')
        plt.title('Hydraulic Erosion Map')
        plt.colorbar(label='Height')
        plt.show()

        # ---- 6. Slope-based erosion feedback (Item 4) ----
        # Cliffs sharpen, valleys smooth, drainage networks emerge naturally.
        # Gradient in world-space [0,1] coords so slope_feedback_strength has a
        # consistent meaning (dh per world-unit) regardless of base grid size.
        #combined = combined.astype(np.float64)
        #grad_x, grad_y = np.gradient(combined, x, y)
        #slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
        #erosion_mask = np.exp(-slope * self.erosion_params.get('slope_feedback_strength', 5.0))
        #combined *= erosion_mask

        plt.imshow(combined, cmap='terrain')
        plt.title('Slope-based Erosion Feedback Map')
        plt.colorbar(label='Height')
        plt.show()

        # ---- 7. FFT low-pass: remove spike artefacts at base-map resolution ----
        # Cutoff and rolloff come from world_params so the same physical wavelength
        # threshold applies whenever the base map is smoothed.
        wp = self.world_params
        #combined = fft_lowpass(
        #    combined,
        #    cutoff=wp['fft_cutoff'],
        #    rolloff=wp['fft_rolloff'],
        #)

        # ---- 8. Percentile normalisation (Item 5) ----
        # Preserves extreme cliffs; avoids washed-out continents.
        #lo, hi = np.percentile(combined, [2, 98])
        #combined = np.clip((combined - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        combined = (combined - combined.min()) / (combined.max() - combined.min())

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
            #scale to [-1, 1] for better control over amplification/suppression via exp-scaling
            noise = noise / (np.abs(noise).max() + 1e-8)
            
            #noise =  np.exp(param['weight'] * noise) #amplify detail in mid-range terrain, suppress at high altitudes
            #heights  *= noise
            weight = param['weight'] / (param['base_freq'])  #scale weight by base_freq to keep the same physical influence regardless of the frequency
            heights = heights * (1 - weight) + noise * weight  #blend rather than pure multiplication to preserve some of the base structure
        
        # ---- Slope-based erosion feedback at fine scale (Item 4) ----
        # Gradient is computed in world-space coordinates [0, 1] so slope is
        # dh/d_world — invariant to zoom level and output resolution.
        # At any zoom, the same physical slope produces the same mask value.
        #grad_x, grad_y = np.gradient(heights, x, y)
        #slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
        #erosion_mask = np.exp(-slope * self.erosion_params.get('slope_feedback_strength', 5.0))
        #heights *= erosion_mask


        wp = self.world_params
        #cell_size_x, cell_size_y = self.get_cell_size(lim, shape)
        #base_cell = wp['max_size'] / self.base_params['size']
        #adaptive_cutoff = float(np.clip(
        #    wp['fft_cutoff'] * base_cell / cell_size_x,
        #    0.05, 0.45,
        #))
        #heights = fft_lowpass(heights, cutoff=adaptive_cutoff, rolloff=wp['fft_rolloff'])

        return heights

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def get_cell_size(self, lim=(0.0, 1.0, 0.0, 1.0), shape=(100, 100)):
        """Get the world-space size of each cell in the output grid in meters."""
        max_range = self.world_params['max_size']
        x_min, x_max, y_min, y_max = lim
        cell_size_x = (x_max - x_min) * max_range / shape[0]
        cell_size_y = (y_max - y_min) * max_range / shape[1]
        return cell_size_x, cell_size_y
    def plot(self, lim=(0.0, 1.0, 0.0, 1.0), shape=(1024, 1024), save_path=None, azim=None, elev=None):
        height_map = self.get_height(lim, shape)
        plot(height_map, self.world_params, lim=lim, save_path=save_path, azim=azim, elev=elev)
    
    def plot2D(self, height_map, ax = None, lim=(0.0, 1.0, 0.0, 1.0)):
        # delegate to the new plotter.plot2D signature which is:
        # plot2D(height_map, world_params, hillshade_map=None, ax=None, lim=...)
        return plot2D(height_map, self.world_params, hillshade_map=None, ax=ax, lim=lim)

    def plot3D(self, height_map, ax = None, lim=(0.0, 1.0, 0.0, 1.0)):
        # delegate to the new plotter.plot3D signature which is:
        # plot3D(height_map, world_params, hillshade_map=None, ax=None, lim=...)
        return plot3D(height_map, self.world_params, hillshade_map=None, ax=ax, lim=lim)
    

        


