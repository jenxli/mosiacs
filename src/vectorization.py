import numpy as np
import cv2 
from scipy.spatial import Delaunay
from voronoi import euclidean_voronoi, chebyshev_voronoi, manhattan_voronoi
import constants

def sample_points_from_image(img, num_points=500, point_color=(0, 0, 255), point_radius=1):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    laplacian_abs = cv2.convertScaleAbs(laplacian)

    threshold_value = 3
    _, thresholded_laplacian = cv2.threshold(laplacian_abs, threshold_value, 255, cv2.THRESH_TOZERO)

    prob_map = thresholded_laplacian / thresholded_laplacian.sum()

    flat_probs = prob_map.ravel()
    indices = np.arange(flat_probs.size)
    chosen_indices = np.random.choice(indices, size=num_points, p=flat_probs)

    coords_y, coords_x = np.unravel_index(chosen_indices, shape=gray.shape)

    coords = np.stack([coords_x, coords_y], axis=1)

    laplacian_color = cv2.merge([thresholded_laplacian] * 3)

    # for visualization purposes
    image_with_points = laplacian_color.copy()
    for x, y in coords:
        cv2.circle(image_with_points, (int(x), int(y)), point_radius, point_color, -1)

    return np.array(coords), image_with_points


def delaunay(img, sampling_method=constants.UNIFORM, faces=None):
    height, width, _ = img.shape

    init_num_points = 200
    points = np.vstack([
        np.random.randint(0, width, size=init_num_points),
        np.random.randint(0, height, size=init_num_points)
    ]).T

    if sampling_method == constants.UNIFORM:
        num_additional_points = 400
        additional_points = np.vstack([
            np.random.randint(0, width, size=num_additional_points),
            np.random.randint(0, height, size=num_additional_points)
        ]).T
        points = np.vstack([points, additional_points])
    elif sampling_method == constants.TARGETED:
        for (face_x, face_y, face_w, face_h) in faces:
            num_points_face = 400

            points_face = np.vstack([
                np.random.randint(face_x, face_x + face_w, size=num_points_face),
                np.random.randint(face_y, face_y + face_h, size=num_points_face)
            ]).T

            points = np.vstack([points, points_face])
    elif sampling_method == constants.PROBABILISTIC:
        points_probabilistic, laplacian = sample_points_from_image(img, num_points=400)
        points = np.vstack([points, points_probabilistic])

    # add image corners
    corners = np.array([[0,0], [0,height-1], [width-1,0], [width-1,height-1]])
    points = np.concatenate([points, corners], axis=0)

    tri = Delaunay(points)

    output = np.zeros_like(img)

    for triangle in tri.simplices:
        pts = points[triangle]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts.astype(np.int32), 1)

        mean_color = cv2.mean(img, mask=mask)[:3]
        mean_color = tuple(map(int, mean_color))

        cv2.fillConvexPoly(output, pts.astype(np.int32), mean_color)
    
    return output

def naive_voronoi(img, metric):
    # sample points along laplacian
    points, laplacian = sample_points_from_image(img, num_points=50)
    width = len(img[0])
    height = len(img)
    voronoi = None

    print("Beginning naive voronoi...")

    # -.- me realizing python 3.9 doesn't have switch statements...    
    if metric == constants.EUCLIDEAN:
        voronoi = euclidean_voronoi(points, width, height)
    elif metric == constants.CHEBYSHEV:
        voronoi = chebyshev_voronoi(points, width, height)
    elif metric == constants.MANHATTAN:
        voronoi = manhattan_voronoi(points, width, height)

    print(" --> Complete!")

    image_out = np.zeros_like(img)

    for i in range(len(points)):
        mask = (voronoi == i)
        if np.any(mask):
            avg_color = img[mask].mean(axis=0)
            image_out[mask] = avg_color

    return image_out, laplacian
