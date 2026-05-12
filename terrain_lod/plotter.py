import matplotlib.pyplot as plt
import numpy as np

plotter_params = {
    'shade_azim': 45,
    'shade_elev': 30,
    '3D_azim': -180-30,
    '3D_elev': 30,
}

def hillshade(height_map, world_params, lim = (0.0, 1.0, 0.0,1.0)):
    #implement hillshade calculation based on terrain gradients and plotter_params['shade_azim'] and plotter_params['shade_elev']

    max_altitude = world_params['max_altitude'] #max altitude in meters
    max_size = world_params['max_size'] #world size in meters

    x_min, x_max, y_min, y_max = lim
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    cell_size = (max_size * max_range) / height_map.shape[0] #assuming square height map

    z = height_map * max_altitude #scale height map to actual altitudes
    # Calculate gradients
    dzdy, dzdx = np.gradient(z, cell_size, cell_size) #calculate gradients in x and y directions
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2)) #calculate slope
    aspect = np.arctan2(-dzdy, dzdx)   

    # Calculate hillshade
    az_rad = np.radians(plotter_params['shade_azim'])
    el_rad = np.radians(plotter_params['shade_elev'])
    az = np.radians(plotter_params['shade_azim'])
    el = np.radians(plotter_params['shade_elev'])

    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect = np.arctan2(-dzdy, dzdx)

    hs = (
        np.sin(el) * np.cos(slope) +
        np.cos(el) * np.sin(slope) * np.cos(az - aspect)
    )

    hillshade = np.clip(255 * hs, 0, 255)
    return hillshade
    
def compute_normals(z, scale=1.0):
    dzdx = np.gradient(z, axis=1)
    dzdy = np.gradient(z, axis=0)

    normals = np.dstack((-dzdx * scale, -dzdy * scale, np.ones_like(z)))
    
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= norm
    return normals

def cheap_ao(z):
    gx, gy = np.gradient(z)
    slope = np.sqrt(gx**2 + gy**2)
    ao = 1.0 / (1.0 + 5.0 * slope)
    return ao

def lambert_shade(z, light_dir=(0.3, 0.3, 1.0)):
    n = compute_normals(z)

    L = np.array(light_dir)
    L = L / np.linalg.norm(L)

    shade = np.clip(
        n[..., 0] * L[0] +
        n[..., 1] * L[1] +
        n[..., 2] * L[2],
        0, 1
    )
    return shade

def plot(height_map, world_params, lim=(0.0, 1.0, 0.0, 1.0),save_path=None, azim = None, elev = None):
    """Convenience method to plot the height map."""
    hillshade_map = hillshade(height_map, world_params, lim)
    #2d hillshaded map with matplotlib, and 3D surface plot side by side
    fig = plt.figure(figsize=(16, 12))
    #make the 2d map smaller than the 3d plot, so the 3d plot is more visible
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    ax1 = fig.add_subplot(gs[0, 0])
    plot2D(height_map, world_params, hillshade_map=hillshade_map, ax=ax1, lim=lim)
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    plot3D(height_map, world_params, hillshade_map=hillshade_map, ax=ax2, lim=lim, azim=azim, elev=elev)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

    plot_slope_histogram(height_map, world_params, lim=lim)


def plot2D(height_map, world_params, hillshade_map=None, ax= None, lim = (0.0, 1.0, 0.0,1.0)):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if hillshade_map is None:
        hillshade_map = hillshade(height_map, world_params, lim)
    
    #plot height map with hillshade as shading
    ax.imshow(height_map, cmap='terrain', extent=lim, origin='lower', vmin=0, vmax=1)
    ax.imshow(hillshade_map, cmap='gray', extent=lim, origin='lower', alpha=0.5) #overlay hillshade with some transparency
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Height Map with Hillshade')
    return ax
    
def terrain_color(z):
    vmax = 1.0
    vmin = 0.0
    # base biome coloring
    colors = plt.cm.terrain(z)[..., :3]  # RGB only

    return colors

def plot3D(height_map, world_params, hillshade_map=None, ax = None, lim = (0.0, 1.0, 0.0,1.0), azim = None, elev = None):
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    if hillshade_map is None:
        hillshade_map = hillshade(height_map, world_params, lim)
    x_min, x_max, y_min, y_max = lim
    x = np.linspace(x_min, x_max, height_map.shape[1])
    y = np.linspace(y_min, y_max, height_map.shape[0])
    x, y = np.meshgrid(x, y, indexing='ij')

    

    h_range = height_map.max() - height_map.min()
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = 1 + 4 * max(x_range, y_range)

    min_lim = height_map.min() - 0.05 * h_range
    max_lim = height_map.max() + 1 * h_range * max_range
    

    shade = lambert_shade(height_map)
    shade = shade * cheap_ao(height_map) #combine lambert shading with ambient occlusion for better visual effect
    shade = np.clip(shade, 0, 1.0)[..., None] #avoid too dark areas

    final_colors = terrain_color(height_map) * shade
    ax.plot_surface(x, y, height_map, linewidth=0, cmap = 'terrain', vmin=0, vmax=1, facecolors=final_colors)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title('3D Terrain')
    ax.set_zlim(min_lim, max_lim)
    
    elev = elev if elev is not None else plotter_params['3D_elev']
    azim = azim if azim is not None else plotter_params['3D_azim']
    ax.view_init(elev=elev, azim=azim)
    return ax

def plot_slope_histogram(height_map, world_params, ax=None, lim=(0.0, 1.0, 0.0, 1.0)):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    x_min, x_max, y_min, y_max = lim
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    cell_size = (world_params['max_size'] * max_range) / height_map.shape[0]

    z = height_map * world_params['max_altitude']
    dzdy, dzdx = np.gradient(z, cell_size, cell_size)
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope = np.degrees(slope) #convert slope to degrees for better interpretability

    ax.hist(slope.flatten(), bins=100)
    ax.set_xlabel('Slope')
    ax.set_ylabel('Frequency')
    ax.set_title('Slope Histogram')
    return ax