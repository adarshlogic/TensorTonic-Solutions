import numpy as np

def apply_homogeneous_transform(T, points):

    T = np.asarray(T)
    points = np.asarray(points)

    single = points.ndim == 1
    if single:
        points = points[None,:]

    points_h = np.hstack([points, np.ones((points.shape[0],1))])

    transformed = points_h @ T.T

    result = transformed[:,:3]

    return result[0] if single else result