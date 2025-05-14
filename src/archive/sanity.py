# Testing broadcasting to make sure it works (just a sanity check): 
import numpy as np

voronoi = np.zeros((5, 7), dtype=int)

sites = np.array([[3, 4], [0, 5], [1, 2]]) # N = 3 sites

sx = sites[:, 0][:, None, None]  
sy = sites[:, 1][:, None, None]  

print(sx.shape)
print(sy)

x = np.arange(7)
y = np.arange(5)
grid_x, grid_y = np.meshgrid(x, y)  
print(grid_x.shape)
print(grid_y)

print((grid_x - sx).shape)
print(grid_y - sy)
