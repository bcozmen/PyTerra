


world_params = {
    "max_size" : 100_000.0,  # world is 100 km across in world-space units
    "max_altitude": 3000.0,   # max elevation is 1000 m in world-space units
    'fft_cutoff': 0.2,       # controls overall terrain smoothness (0-1)
    'fft_rolloff': 1.0,       # controls sharpness of FFT cutoff
}
#: Low-resolution base grid (diamond-square).  size must be 2^n + 1.
base_params = {
    'size': 2**12 + 1,
    'scale': 1.0,
    'roughness': 0.55,
}


#: Controls every noise layer and the erosion passes.
erosion_params = {
    'thermal_iterations': 50,
    'talus': 0.02,
    'hydraulic_iterations_density': 1.0,
    'erosion_rate': 0.03,
    'deposition_rate': 0.015,
    'evaporation': 0.015,
    'slope_feedback_strength' : 0.0015
}

continent_params = {
    'octaves': 3,
    'persistence': 0.5,
    'lacunarity': 2.0,
    'base_freq': 0.2,      # very low -> broad ocean/continent separation
    'land_fraction': 0.75,  # bias: fraction of world that is above water (0-1)
}

macro_params1 = {
    'octaves' : 6, 
    'persistence': 0.5,
    'lacunarity': 2.0,
    'base_freq': 0.5,       # very low -> large blobs
    'weight': 0.15,        # fraction of final height driven by macro layer
    'warp_x': 1.2,        # no warp on macro layer -- pure continental blobs
    'warp_y': 1.2,
    'ridge_strength': 0.0,   # no ridges on macro layer -- pure FBM hills
}

macro_params = {
    'octaves' : 6, 
    'persistence': 0.5,
    'lacunarity': 2.0,
    'base_freq': 2.0,       # very low -> large blobs
    'weight': 0.3,        # fraction of final height driven by macro layer
    'warp_x': 0.8,        # no warp on macro layer -- pure continental blobs
    'warp_y': 1.0,
    'ridge_strength': 0.15,   # no ridges on macro layer -- pure FBM hills
}


mid_params = {
    'octaves': 6,
    'persistence': 0.4,
    'lacunarity': 2.0,
    'base_freq': 4.0,
    'weight': 0.25,
    'warp_x': 0.5,      # anisotropic warp -- larger -> more elongated ridges
    'warp_y': 0.2,
    'ridge_strength': 0.3, # 0 = plain FBM hills, 1 = pure ridge noise
}

meso_params = {
    'octaves': 6,
    'persistence': 0.4,
    'lacunarity': 2.0,
    'base_freq': 10.0,   # << IMPORTANT: fills mid band
    'weight': 0.15,
    'warp_x': 0.25,
    'warp_y': 0.15,
    'ridge_strength': 0.15,
}

meso2_params = {
    'octaves': 6,
    'persistence': 0.4,
    'lacunarity': 2.0,
    'base_freq': 20.0,   # << IMPORTANT: fills mid band
    'weight': 0.10,
    'warp_x': 0.07,
    'warp_y': 0.1,
    'ridge_strength': 0.05,
}

meso3_params = {
    'octaves': 6,
    'persistence': 0.4,
    'lacunarity': 2.0,
    'base_freq': 40.0,   # << IMPORTANT: fills mid band
    'weight': 0.05,
    'warp_x': 0.02,
    'warp_y': 0.4,
    'ridge_strength': 0.05,
}

micro_params = {   
    'octaves': 3,
    'persistence': 0.5,
    'lacunarity': 2.0,
    'base_freq': 60.0,       # high freq -> rock / surface variation
    'weight': 0.01,      # kept very small so base structure dominates
    'warp_x': 0.05,        # subtle domain warp on micro layer for natural variation
    'warp_y': 0.05,
    'ridge_strength': 0.01,   # some ridge character in micro layer for sharper rock detail
}




base_noise_params = [macro_params1,macro_params, mid_params, meso_params, meso2_params, meso3_params]
micro_noise_params = [micro_params]