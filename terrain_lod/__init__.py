"""Public package interface for the terrain_lod module.

Expose the main `Terrain` class and the default parameter bundles so callers
can `from terrain_lod import Terrain` or access the defaults directly.
"""

from .terrain import (
	Terrain,
	base_params,
	erosion_params,
	base_noise_params,
	micro_noise_params,
	continent_params,
)

__all__ = [
	"Terrain",
	"base_params",
	"erosion_params",
	"base_noise_params",
	"micro_noise_params",
	"continent_params",
]
