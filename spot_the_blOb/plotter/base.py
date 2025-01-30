import matplotlib.pyplot as plt
import cartopy.feature as cfeature

class PlotterBase:
    def __init__(self, xarray_obj):
        self.da = xarray_obj
        
        # Cache common features
        self._land = cfeature.LAND.with_scale('50m')
        self._coastlines = cfeature.COASTLINE.with_scale('50m')
    
    def clim_robust(self, data, issym, percentiles=[2, 98]):
        """Base method for computing color limits"""
        clim = np.nanpercentile(data, percentiles)
        
        if issym:
            clim = np.abs(clim).max()
            clim = np.array([-clim, clim])
        elif percentiles[0] == 0:
            clim = np.array([0, clim[1]])

        return clim
    
    def setup_plot_params(self):
        """Setup common plotting parameters"""
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')