# ---------------------------------------------------------------------------
# Default parameter bundles
# ---------------------------------------------------------------------------

world_params = {
    "max_size" : 100_000.0,  # world is 100 km across in world-space units
    "max_altitude": 1000.0,   # max elevation is 1000 m in world-space units
    'fft_cutoff': 0.1,       # controls overall terrain smoothness (0-1)
    'fft_rolloff': 5.0,       # controls sharpness of FFT cutoff
}
#: Low-resolution base grid (diamond-square).  size must be 2^n + 1.
base_params = {
    'size': 2**10 + 1,
    'scale': 1.0,
    'roughness': 0.55,
}


#: Controls every noise layer and the erosion passes.
erosion_params = {
    'thermal_iterations': 250,
    'talus': 0.025,
    'hydraulic_iterations': 500,
    'erosion_rate': 0.03,
    'deposition_rate': 0.03,
    'evaporation': 0.02,
}

continent_params = {
    'octaves': 3,
    'persistence': 0.5,
    'lacunarity': 2.0,
    'base_freq': 0.2,      # very low -> broad ocean/continent separation
    'land_fraction': 0.75,  # bias: fraction of world that is above water (0-1)
}

macro_params = {
    'octaves' : 3, 
    'persistence': 0.5,
    'lacunarity': 2.0,
    'base_freq': 1.0,       # very low -> large blobs
    'weight': 0.30,        # fraction of final height driven by macro layer
    'warp_x': 1.2,        # no warp on macro layer -- pure continental blobs
    'warp_y': 0.8,
    'ridge_strength': 0.0,   # no ridges on macro layer -- pure FBM hills
}


mid_params = {
    'octaves': 4,
    'persistence': 0.5,
    'lacunarity': 2.0,
    'base_freq': 4.0,
    'weight': 0.35,
    'warp_x': 1.25,      # anisotropic warp -- larger -> more elongated ridges
    'warp_y': 0.25,
    'ridge_strength': 0.1, # 0 = plain FBM hills, 1 = pure ridge noise
}

meso_params = {
    'octaves': 2,
    'persistence': 0.5,
    'lacunarity': 2.0,
    'base_freq': 64.0,   # << IMPORTANT: fills mid band
    'weight': 0.1,
    'warp_x': 0.8,
    'warp_y': 1.2,
    'ridge_strength': 0.1,
}

micro_params = {   
    'octaves': 8,
    'persistence': 0.5,
    'lacunarity': 2.0,
    'base_freq': 100.0,       # high freq -> rock / surface variation
    'amplitude': 0.1,      # kept very small so base structure dominates
    'warp_x': 0.1,        # subtle domain warp on micro layer for natural variation
    'warp_y': 0.05,
    'ridge_strength': 0.25,   # some ridge character in micro layer for sharper rock detail
}


base_noise_params = [macro_params, mid_params, meso_params]
micro_noise_params = [micro_params]
