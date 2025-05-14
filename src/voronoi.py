import numpy as np

def manhattan_voronoi(sites, width, height):
    voronoi = np.zeros((height, width), dtype=int)
    sx = sites[:, 0][:, None, None]  
    sy = sites[:, 1][:, None, None]  

    x = np.arange(width)
    y = np.arange(height)
    grid_x, grid_y = np.meshgrid(x, y)  

    dist_manhattan = np.abs(grid_x - sx) + np.abs(grid_y - sy)
    voronoi = np.argmin(dist_manhattan, axis=0)  
    return voronoi

def chebyshev_voronoi(sites, width, height):
    voronoi = np.zeros((height, width), dtype=int)
    sx = sites[:, 0][:, None, None]  
    sy = sites[:, 1][:, None, None]  

    x = np.arange(width)
    y = np.arange(height)
    grid_x, grid_y = np.meshgrid(x, y)  

    dist_chebyshev = np.maximum(np.abs(grid_x - sx), np.abs(grid_y - sy))
    voronoi = np.argmin(dist_chebyshev, axis=0)  
    return voronoi

def euclidean_voronoi(sites, width, height):
    voronoi = np.zeros((height, width), dtype=int)
    sx = sites[:, 0][:, None, None]  
    sy = sites[:, 1][:, None, None] 

    x = np.arange(width)
    y = np.arange(height)
    grid_x, grid_y = np.meshgrid(x, y) 

    dist_squared = (grid_x - sx)**2 + (grid_y - sy)**2  
    voronoi = np.argmin(dist_squared, axis=0)  
    return voronoi

