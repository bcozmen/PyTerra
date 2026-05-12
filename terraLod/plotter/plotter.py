import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np

from ..helper import normalize, get_grid
from .shade import get_3D_shade, hillshade
from .helper import get_cell_size, set_labels, find_z_limits




class Plotter():
    def __init__(self, plotter_params):
        for key in plotter_params:
            setattr(self, key, plotter_params.get(key, plotter_params[key]))
    def plot(self, height_map, lim=(0.0, 1.0, 0.0, 1.0), save_path=None, shade = True, plot_slope_histogram = True):
        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.05)

        z = height_map * self.max_altitude
        cell_size, max_range = get_cell_size(lim, self.max_size, height_map.shape)
        dzdy, dzdx = np.gradient(z, cell_size[1], cell_size[0])
        gradients = (dzdy, dzdx)

        ax1 = fig.add_subplot(gs[0, 0])
        self.plot2D(height_map, gradients, ax=ax1, lim=lim, shade=shade)

        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        self.plot3D(height_map, gradients, ax=ax2, lim=lim, shade=shade)

        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        if plot_slope_histogram:
            self.plot_slope_histogram(height_map, gradients, lim=lim)
    def plot3D(self, height_map, gradients, ax = None, lim = (0.0, 1.0, 0.0,1.0), shade = True):
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        x,y = get_grid(lim=lim, shape=height_map.shape)
        
        cell_size, max_range = get_cell_size(lim, self.max_size, height_map.shape)
        z_meters = height_map * self.max_altitude

        base_color = plt.cm.terrain(height_map)[..., :3]  # RGB only
        if shade:
            shade = get_3D_shade(self, height_map, gradients, lim, self.max_size, self.max_altitude, self.ambient)
            base_color = base_color * shade

        ax.plot_surface(x, y, height_map, facecolors=base_color, linewidth=0, antialiased=True)
        
        z_lim = find_z_limits(height_map, lim, height_map.shape, self.max_size)
        set_labels(ax, zlim=z_lim, z_label='Height (normalized)', title='3D Terrain')
        ax.view_init(elev=self.elev, azim=self.azim)
        
        #set axis limits to match 2D plot
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        return ax


    def plot2D(self, height_map, gradients, ax= None, lim = (0.0, 1.0, 0.0,1.0), shade = True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(height_map, cmap='terrain', extent=lim, origin='lower', vmin=0, vmax=1)
        
        if shade:
            hillshade_map = hillshade(height_map, gradients, lim, self.max_altitude, self.max_size, self.shade_azim, self.shade_elev)
            ax.imshow(hillshade_map, cmap='gray', extent=lim, origin='lower', alpha=0.5) #overlay hillshade with some transparency
        set_labels(ax, z_label=None, title='Height Map with Hillshade')
        return ax

    def plot_slope_histogram(self, height_map, gradients, lim=(0.0, 1.0, 0.0, 1.0), ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        cell_size, max_range = get_cell_size(lim, self.max_size, height_map.shape)
        
        z = height_map * self.max_altitude
        dzdy, dzdx = gradients
        slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        slope = np.degrees(slope) #convert slope to degrees for better interpretability

        ax.hist(slope.flatten(), bins=100)
        set_labels(ax, x_label='Slope (degrees)', y_label='Frequency', title='Slope Histogram', z_label=None)
        return ax

