import numpy as np 
import matplotlib.pyplot as plt 

from utils import plot_slope_intercept_line
from pathlib import Path


if __name__ == '__main__':
    
    x = np.linspace(-5, 5, 100)
    
    # line 1
    params = {
        'axs': None,
        'color': 'b',
        'linestyle': 'solid',
        'linewidth': 2        
    }
    slope, intercept = 4/3, 0
    axs = plot_slope_intercept_line(x, slope, intercept, **params)
    
    # line 2
    params.update({'axs': axs, 'color': 'r'})
    slope, intercept = -2/3, 6
    axs = plot_slope_intercept_line(x, slope, intercept, **params)
    
    axs.grid('on')
    axs.set_aspect('equal')
    axs.set_xlabel('$x_{1}$')
    axs.set_ylabel('$x_{2}$')    
    
    # output path
    output_path = Path('./figs/intersection_two_lines.jpg')
    plt.savefig(output_path)
    
    plt.show()
                       