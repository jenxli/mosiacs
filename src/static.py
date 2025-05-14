import cv2 
import numpy as np
from vectorization import naive_voronoi
import constants
import matplotlib.pyplot as plt

OUT_DIR = './src/data/portrait-color-male-p50/'
IMG_PATH = OUT_DIR + 'portrait-color-male.jpg'

img_bgr = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)    

mosaic_euclidean, laplacian_euclidean = naive_voronoi(img_rgb, constants.EUCLIDEAN)
plt.imsave(OUT_DIR + 'euclidean_voronoi_out.jpg', mosaic_euclidean)
plt.imsave(OUT_DIR + 'euclidean_sampling_out.jpg', laplacian_euclidean)

mosaic_chebyshev, laplacian_chebyshev = naive_voronoi(img_rgb, constants.CHEBYSHEV)
plt.imsave(OUT_DIR + 'chebyshev_voronoi_out.jpg', mosaic_chebyshev)
plt.imsave(OUT_DIR + 'chebyshev_sampling_out.jpg', laplacian_chebyshev)

mosaic_manhattan, laplacian_manhattan = naive_voronoi(img_rgb, constants.MANHATTAN)
plt.imsave(OUT_DIR + 'manhattan_voronoi_out.jpg', mosaic_manhattan)
plt.imsave(OUT_DIR + 'manhattan_sampling_out.jpg', laplacian_manhattan)


